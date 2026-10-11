package objectstore

import (
	"context"
	"errors"
	"net/http"
)

// Delete removes the object at key. A missing object is not an error.
func (c *Client) Delete(ctx context.Context, key string) error {
	u, host := c.objectURL(key)

	resp, err := c.doRequest(ctx, "DELETE", u, host, 0, nil)
	if err != nil {
		var apiErr *APIError
		if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusNotFound {
			return nil
		}
		return err
	}
	_ = resp.Body.Close()
	return nil
}
