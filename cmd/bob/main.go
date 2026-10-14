package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/mattn/go-isatty"

	"bob/internal/bootstrap"
	"bob/internal/cli"
	"bob/internal/config"
	"bob/internal/sandbox"
)

type stringSliceFlag []string

func (s *stringSliceFlag) String() string {
	return strings.Join(*s, ",")
}

func (s *stringSliceFlag) Set(v string) error {
	if strings.TrimSpace(v) == "" {
		return errors.New("flag value cannot be empty")
	}
	for _, part := range strings.Split(v, ",") {
		trimmed := strings.TrimSpace(part)
		if trimmed == "" {
			return errors.New("flag value contains empty element")
		}
		*s = append(*s, trimmed)
	}
	return nil
}

type runtimeFactoryFunc func(cfg *config.Config, opts bootstrap.Options) (*bootstrap.Runtime, error)

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	return runWithFactory(args, stdin, stdout, stderr, bootstrap.New)
}

func runWithFactory(args []string, stdin io.Reader, stdout, stderr io.Writer, factory runtimeFactoryFunc) int {
	var (
		promptVal         string
		quietVal          bool
		verboseVal        bool
		initConfigVal     bool
		purgeCachesVal    bool
		configVal         string
		mountVal          string
		sandboxVal        string
		netVal            string
		imageVal          string
		modelVal          string
		maxIterVal        int
		timeoutVal        time.Duration
		allowDomains      stringSliceFlag
		extraPaths        stringSliceFlag
		autoToolchainsVal bool
	)

	fs := flag.NewFlagSet("bob", flag.ContinueOnError)
	fs.SetOutput(stderr)

	fs.StringVar(&promptVal, "p", "", "Prompt to execute")
	fs.StringVar(&promptVal, "prompt", "", "Prompt to execute")
	fs.BoolVar(&quietVal, "q", false, "Quiet mode; suppress intermediate steps")
	fs.BoolVar(&quietVal, "quiet", false, "Quiet mode; suppress intermediate steps")
	fs.BoolVar(&verboseVal, "v", false, "Verbose debug logging on stderr")
	fs.BoolVar(&verboseVal, "verbose", false, "Verbose debug logging on stderr")
	fs.BoolVar(&initConfigVal, "init-config", false, "Initialize config template and exit")
	fs.BoolVar(&purgeCachesVal, "purge-caches", false, "Purge all sandbox cache directories and exit")
	fs.StringVar(&configVal, "config", "", "Path to configuration file")
	fs.StringVar(&mountVal, "mount", "", "Workspace mount spec (PATH[:ro|:rw])")
	fs.StringVar(&sandboxVal, "sandbox", "", "Sandbox driver ('bwrap' or 'docker')")
	fs.StringVar(&sandboxVal, "driver", "", "Sandbox driver alias ('bwrap' or 'docker')")
	fs.StringVar(&netVal, "net", "", "Sandbox network mode ('none', 'restricted', 'full')")
	fs.StringVar(&imageVal, "image", "", "Docker container image")
	fs.StringVar(&modelVal, "model", "", "LLM model override")
	fs.IntVar(&maxIterVal, "max-iterations", 0, "Maximum tool iterations (default 20)")
	fs.DurationVar(&timeoutVal, "timeout", 1*time.Hour, "Overall execution timeout")
	fs.Var(&allowDomains, "allow-domain", "Allowed domain in restricted network mode (repeatable)")
	fs.Var(&extraPaths, "extra-path", "Extra host directory to mount read-only and add to sandbox $PATH (repeatable)")
	fs.BoolVar(&autoToolchainsVal, "auto-toolchains", false, "Automatically discover host toolchains (e.g. mise, pyenv, cargo) and mount read-only")
	fs.BoolVar(&autoToolchainsVal, "discover-toolchains", false, "Alias for --auto-toolchains")

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	if len(fs.Args()) > 0 {
		_, _ = fmt.Fprintf(stderr, "Error: unexpected argument %q\n", fs.Args()[0])
		return 2
	}

	visited := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "p", "prompt":
			visited["prompt"] = true
		case "q", "quiet":
			visited["quiet"] = true
		case "v", "verbose":
			visited["verbose"] = true
		case "init-config":
			visited["init-config"] = true
		case "purge-caches":
			visited["purge-caches"] = true
		case "config":
			visited["config"] = true
		case "mount":
			visited["mount"] = true
		case "sandbox", "driver":
			visited["sandbox"] = true
		case "net":
			visited["net"] = true
		case "image":
			visited["image"] = true
		case "model":
			visited["model"] = true
		case "max-iterations":
			visited["max-iterations"] = true
		case "timeout":
			visited["timeout"] = true
		case "allow-domain":
			visited["allow-domain"] = true
		case "extra-path":
			visited["extra-path"] = true
		case "auto-toolchains", "discover-toolchains":
			visited["auto-toolchains"] = true
		}
	})

	// 1. Handle --purge-caches
	if visited["purge-caches"] && purgeCachesVal {
		count, err := sandbox.PurgeAllCaches()
		if err != nil {
			_, _ = fmt.Fprintf(stderr, "Error purging sandbox caches: %v\n", err)
			return 1
		}
		_, _ = fmt.Fprintf(stdout, "Purged %d cache(s).\n", count)
		return 0
	}

	// 2. Handle --init-config
	if visited["init-config"] {
		targetPath := configVal
		if !visited["config"] || targetPath == "" {
			defaultCfg, _, _ := config.DefaultPaths()
			targetPath = defaultCfg
		}
		if err := config.WriteTemplate(targetPath); err != nil {
			_, _ = fmt.Fprintf(stderr, "Error writing config template: %v\n", err)
			return 1
		}
		_, _ = fmt.Fprintf(stdout, "Initialized configuration template at %s\n", targetPath)
		return 0
	}

	// 3. Validate prompt presence
	if !visited["prompt"] {
		_, _ = fmt.Fprintln(stderr, "Usage: bob -p <prompt> [options]")
		_, _ = fmt.Fprintln(stderr, "Error: interactive mode is not implemented yet; provide -p <prompt>")
		return 2
	}
	if strings.TrimSpace(promptVal) == "" && promptVal != "-" {
		_, _ = fmt.Fprintln(stderr, "Error: prompt cannot be empty")
		return 2
	}

	// 3. Validate flag combinations before loading config
	if visited["mount"] && strings.TrimSpace(mountVal) == "" {
		_, _ = fmt.Fprintln(stderr, "Error: --mount path cannot be empty")
		return 2
	}
	if !visited["mount"] {
		if visited["sandbox"] {
			_, _ = fmt.Fprintln(stderr, "Error: --sandbox requires --mount")
			return 2
		}
		if visited["net"] {
			_, _ = fmt.Fprintln(stderr, "Error: --net requires --mount")
			return 2
		}
		if visited["image"] {
			_, _ = fmt.Fprintln(stderr, "Error: --image requires --mount")
			return 2
		}
	}
	if visited["allow-domain"] && (!visited["net"] || netVal != "restricted") {
		_, _ = fmt.Fprintln(stderr, "Error: --allow-domain requires --net restricted")
		return 2
	}
	if visited["sandbox"] && sandboxVal != "bwrap" && sandboxVal != "docker" {
		_, _ = fmt.Fprintf(stderr, "Error: invalid sandbox driver %q (must be 'bwrap' or 'docker')\n", sandboxVal)
		return 2
	}
	if visited["net"] && netVal != "none" && netVal != "restricted" && netVal != "full" {
		_, _ = fmt.Fprintf(stderr, "Error: invalid network mode %q (must be 'none', 'restricted', or 'full')\n", netVal)
		return 2
	}
	if visited["max-iterations"] && maxIterVal <= 0 {
		_, _ = fmt.Fprintln(stderr, "Error: --max-iterations must be greater than 0")
		return 2
	}
	if visited["timeout"] && timeoutVal <= 0 {
		_, _ = fmt.Fprintln(stderr, "Error: --timeout must be greater than 0")
		return 2
	}
	if visited["model"] && strings.TrimSpace(modelVal) == "" {
		_, _ = fmt.Fprintln(stderr, "Error: --model cannot be empty")
		return 2
	}

	// 4. Resolve and load configuration
	cfgPath := configVal
	if !visited["config"] || cfgPath == "" {
		defaultCfg, _, _ := config.DefaultPaths()
		cfgPath = defaultCfg
	}

	cfg, err := config.LoadFromFile(cfgPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			_, _ = fmt.Fprintf(stderr, "Error: configuration file not found at %s. Run 'bob --init-config' to create one.\n", cfgPath)
		} else {
			_, _ = fmt.Fprintf(stderr, "Error loading configuration: %v\n", err)
		}
		return 2
	}

	if err := cfg.ValidateCLI(); err != nil {
		_, _ = fmt.Fprintf(stderr, "Error: invalid configuration: %v\n", err)
		return 2
	}

	// Apply CLI overrides to configuration
	if visited["model"] {
		cfg.OpenAIModel = modelVal
	}
	if visited["max-iterations"] {
		cfg.CLIMaxIterations = maxIterVal
	}

	effectiveDriver := cfg.SandboxDefaultDriver
	if effectiveDriver == "" {
		effectiveDriver = "bwrap"
	}
	if visited["sandbox"] {
		effectiveDriver = sandboxVal
	}

	if visited["image"] && effectiveDriver != "docker" {
		_, _ = fmt.Fprintln(stderr, "Error: --image requires docker driver")
		return 2
	}

	// 5. Setup Logging
	logLevel := slog.LevelWarn
	if verboseVal {
		logLevel = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(stderr, &slog.HandlerOptions{Level: logLevel}))
	slog.SetDefault(logger)

	// 6. Signal handling and execution timeout
	sigCtx, stopSignal := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignal()

	go func() {
		<-sigCtx.Done()
		stopSignal() // Unregister signal handler so a second signal terminates process immediately
	}()

	ctx, cancelTimeout := context.WithTimeout(sigCtx, timeoutVal)
	defer cancelTimeout()

	// 7. Parse Mount Spec if requested
	if visited["extra-path"] && !visited["mount"] {
		_, _ = fmt.Fprintln(stderr, "Error: --extra-path requires --mount (workspace must be mounted)")
		return 2
	}
	if visited["auto-toolchains"] && !visited["mount"] {
		_, _ = fmt.Fprintln(stderr, "Error: --auto-toolchains requires --mount (workspace must be mounted)")
		return 2
	}

	var sbSpec *cli.SandboxSpec
	if visited["mount"] {
		cwd, err := os.Getwd()
		if err != nil {
			_, _ = fmt.Fprintf(stderr, "Error resolving current working directory: %v\n", err)
			return 1
		}

		expCfgPath, err := config.ExpandPath(cfgPath)
		if err != nil {
			expCfgPath = cfgPath
		}
		realCfgPath, err := filepath.EvalSymlinks(expCfgPath)
		if err != nil {
			realCfgPath = expCfgPath
		}
		cfgDir := filepath.Dir(expCfgPath)
		realCfgDir := filepath.Dir(realCfgPath)
		realDataDir, err := filepath.EvalSymlinks(cfg.DataDir)
		if err != nil {
			realDataDir = cfg.DataDir
		}
		protected := append(sandbox.DefaultSensitivePaths(), cfgDir, realCfgDir, cfg.DataDir, realDataDir)

		mountSpec, err := cli.ParseMountSpec(mountVal, cwd, protected)
		if err != nil {
			_, _ = fmt.Fprintf(stderr, "Error: invalid mount specification: %v\n", err)
			return 2
		}

		effectiveNet := "none"
		if visited["net"] {
			effectiveNet = netVal
		}
		effectiveImage := cfg.SandboxDockerImage
		if visited["image"] {
			effectiveImage = imageVal
		}

		var combinedExtraPaths []string
		seenPaths := make(map[string]bool)
		for _, p := range cfg.SandboxExtraPaths {
			if !seenPaths[p] {
				seenPaths[p] = true
				combinedExtraPaths = append(combinedExtraPaths, p)
			}
		}
		if visited["extra-path"] {
			for _, p := range extraPaths {
				if !seenPaths[p] {
					seenPaths[p] = true
					combinedExtraPaths = append(combinedExtraPaths, p)
				}
			}
		}

		effectiveAutoToolchains := cfg.SandboxAutoToolchains
		if visited["auto-toolchains"] {
			effectiveAutoToolchains = autoToolchainsVal
		}
		if effectiveAutoToolchains {
			discovered := sandbox.DiscoverToolchains("", cwd, protected)
			for _, p := range discovered {
				if !seenPaths[p] {
					seenPaths[p] = true
					combinedExtraPaths = append(combinedExtraPaths, p)
				}
			}
		}

		for _, p := range combinedExtraPaths {
			if _, err := sandbox.ResolveExtraPath(p, cwd, protected); err != nil {
				_, _ = fmt.Fprintf(stderr, "Error: invalid extra path %q: %v\n", p, err)
				return 2
			}
		}

		sbSpec = &cli.SandboxSpec{
			WorkspaceDir:   mountSpec.HostPath,
			ReadOnly:       mountSpec.ReadOnly,
			Driver:         effectiveDriver,
			DockerImage:    effectiveImage,
			NetworkMode:    effectiveNet,
			AllowedDomains: allowDomains,
			ExtraPaths:     combinedExtraPaths,
			ProtectedPaths: protected,
		}
	}

	// 8. Handle Stdin
	var opts cli.Options
	opts.Quiet = quietVal
	opts.Sandbox = sbSpec
	opts.Model = modelVal
	opts.MaxIterations = maxIterVal
	opts.Timeout = timeoutVal

	if promptVal == "-" {
		if stdin == nil {
			_, _ = fmt.Fprintln(stderr, "Error: stdin is not available")
			return 1
		}
		data, err := readAllWithContext(ctx, stdin, cfg.MaxAttachmentSizeBytes+1)
		if err != nil {
			if errors.Is(err, context.Canceled) || (sigCtx.Err() != nil && errors.Is(sigCtx.Err(), context.Canceled)) {
				return 130
			}
			if errors.Is(err, context.DeadlineExceeded) || (ctx.Err() != nil && errors.Is(ctx.Err(), context.DeadlineExceeded)) {
				_, _ = fmt.Fprintln(stderr, "Error: timed out reading stdin")
				return 1
			}
			_, _ = fmt.Fprintf(stderr, "Error reading stdin: %v\n", err)
			return 1
		}
		if int64(len(data)) > cfg.MaxAttachmentSizeBytes {
			_, _ = fmt.Fprintf(stderr, "Error: stdin exceeds maximum allowed size (%d bytes)\n", cfg.MaxAttachmentSizeBytes)
			return 2
		}
		stdinStr := string(data)
		if strings.TrimSpace(stdinStr) == "" {
			_, _ = fmt.Fprintln(stderr, "Error: prompt from stdin cannot be empty")
			return 2
		}
		opts.Prompt = stdinStr
		opts.Stdin = ""
	} else {
		opts.Prompt = promptVal
		isTTY := false
		if f, ok := stdin.(*os.File); ok {
			isTTY = isatty.IsTerminal(f.Fd()) || isatty.IsCygwinTerminal(f.Fd())
		}
		if !isTTY && stdin != nil {
			data, err := readAllWithContext(ctx, stdin, cfg.MaxAttachmentSizeBytes+1)
			if err != nil {
				if errors.Is(err, context.Canceled) || (sigCtx.Err() != nil && errors.Is(sigCtx.Err(), context.Canceled)) {
					return 130
				}
				if errors.Is(err, context.DeadlineExceeded) || (ctx.Err() != nil && errors.Is(ctx.Err(), context.DeadlineExceeded)) {
					_, _ = fmt.Fprintln(stderr, "Error: timed out reading stdin")
					return 1
				}
				_, _ = fmt.Fprintf(stderr, "Error reading stdin: %v\n", err)
				return 1
			}
			if int64(len(data)) > cfg.MaxAttachmentSizeBytes {
				_, _ = fmt.Fprintf(stderr, "Error: stdin exceeds maximum allowed size (%d bytes)\n", cfg.MaxAttachmentSizeBytes)
				return 2
			}
			opts.Stdin = string(data)
		}
	}

	// 9. Runtime creation and Turn execution
	enableSandbox := visited["mount"]
	rt, err := factory(cfg, bootstrap.Options{
		EnableSandbox: &enableSandbox,
	})
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "Error initializing runtime: %v\n", err)
		return 1
	}

	var closeOnce sync.Once
	var closeErr error
	closeRuntime := func() error {
		closeOnce.Do(func() {
			closeErr = rt.Close()
			if closeErr != nil {
				slog.Warn("error closing runtime", "error", closeErr)
			}
		})
		return closeErr
	}
	defer func() {
		_ = closeRuntime()
	}()

	exitCode := cli.Run(ctx, rt, opts, stdout, stderr)
	if err := closeRuntime(); err != nil {
		_, _ = fmt.Fprintf(stderr, "Error closing runtime: %v\n", err)
		if exitCode == 0 {
			exitCode = 1
		}
	}
	if sigCtx.Err() != nil && errors.Is(sigCtx.Err(), context.Canceled) {
		return 130
	}
	if ctx.Err() != nil && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return 1
	}
	return exitCode
}

func readAllWithContext(ctx context.Context, r io.Reader, limit int64) ([]byte, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	type readResult struct {
		data []byte
		err  error
	}
	ch := make(chan readResult, 1)
	go func() {
		lr := io.LimitReader(r, limit)
		data, err := io.ReadAll(lr)
		ch <- readResult{data: data, err: err}
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case res := <-ch:
		return res.data, res.err
	}
}
