import torch


def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde
    
    Returns:
    --------
        spline values : 3D torch.tensor
            shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.
      
    Example
    -------
    >>> from kan.spline import B_batch
    >>> x = torch.rand(100,2)
    >>> grid = torch.linspace(-1,1,steps=11)[None, :].expand(2, 11)
    >>> B_batch(x, grid, k=3).shape
    '''
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value



def coef2curve(x_eval, grid, coef, k, device="cpu"):
    '''
    Convert B-spline coefficients to B-spline curves, with linear extrapolation
    on the left and convex quadratic extrapolation on the right.

    Args:
        x_eval : Tensor of shape [batch, in_dim]
        grid   : Tensor of shape [in_dim, num_grid_points]
        coef   : Tensor of shape [in_dim, out_dim, n_coef]
        k      : Spline order.
        device : Device (default "cpu").

    Returns:
        y_eval      : Tensor of shape [batch, in_dim, out_dim] (evaluated with extrapolation).
        coef_convex : Modified coefficients (same shape as coef).
    '''
    # Prepare convex coefficients
    c1 = coef[:, :, 0:1]
    s = torch.nn.functional.relu(coef[:, :, 1:])
    d = torch.cat((torch.zeros((coef.shape[0], coef.shape[1], 1), device=device),
                   torch.cumsum(s, dim=2)), dim=2)
    coef_convex = c1.repeat(1, 1, coef.shape[2]) + torch.cumsum(d, dim=2)
    
    # Evaluate the B-spline curve at x_eval
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef_convex.to(b_splines.device))
    
    # Set a small epsilon for local polynomial fitting
    eps = 1e-3
    
    # --- Left endpoint: linear extrapolation ---
    # Use two points near the left boundary: grid[:, k]-eps and grid[:, k]+eps.
    x_poly_left = torch.stack((grid[:, k], grid[:, k]+eps), dim=0)  # Shape: [2, in_dim]
    b_splines_left = B_batch(x_poly_left, grid, k=k)
    y_poly_left = torch.einsum('ijk,jlk->ijl', b_splines_left, coef_convex.to(b_splines_left.device))
    # y_poly_left: [2, in_dim, out_dim]
    # Use these to compute a linear slope:
    y_left_0 = y_poly_left[0, :, :]  # at grid[:, k]-eps
    y_left_1 = y_poly_left[1, :, :]  # at grid[:, k]+eps
    b_poly_left = (y_left_1 - y_left_0) / (eps)

    y_left = y_left_0
    
    # --- Right endpoint: linear extrapolation ---
    # Use three points near the right boundary: grid[:, -k-1]-eps, grid[:, -k-1], grid[:, -k-1]+eps.
    x_poly_right = torch.stack((grid[:, -k-1]-eps, grid[:, -k-1]), dim=0)  # [3, in_dim]
    b_splines_right = B_batch(x_poly_right, grid, k=k)
    y_poly_right = torch.einsum('ijk,jlk->ijl', b_splines_right, coef_convex.to(b_splines_right.device))
    # y_poly_right: [3, in_dim, out_dim]
    y_right_0 = y_poly_right[0, :, :]
    y_right_1 = y_poly_right[1, :, :]
    b_poly_right = (y_right_1 - y_right_0) / (eps)
    y_right = y_right_1

    # --- Extrapolation ---
    # Determine the left and right boundaries from the grid.
    x_min = grid[:, k][:, None]      # [in_dim, 1]
    x_max = grid[:, -k-1][:, None]     # [in_dim, 1]
    
    # Expand x_eval: [batch, in_dim] -> [batch, in_dim, 1] for broadcasting.
    x_eval_exp = x_eval.unsqueeze(2)
    
    # Left extrapolation (linear):  y(x) = y_left  + b_poly_left  * (x - x_min)
    extrap_left = y_left.unsqueeze(0) + b_poly_left.unsqueeze(0) * (x_eval_exp - x_min.unsqueeze(0))
    
    # Right extrapolation (linear): y(x) = y_right + b_poly_right * (x - x_max)
    extrap_right = y_right.unsqueeze(0) + b_poly_right.unsqueeze(0) * (x_eval_exp - x_max.unsqueeze(0))
    
    # Replace values outside the defined grid interval.
    x_min_exp = x_min.unsqueeze(0)
    x_max_exp = x_max.unsqueeze(0)
    
    y_eval = torch.where(x_eval_exp < x_min_exp, extrap_left, y_eval)
    y_eval = torch.where(x_eval_exp > x_max_exp, extrap_right, y_eval)
    
    return y_eval, coef_convex


def curve2coef(x_eval, y_eval, grid, k, lamb=1e-8): # not used in the default settings (i.e., grid adjustments occur only during initialization)
    '''
    Convert B-spline curves to coefficients using least squares,
    preserving endpoint slopes.

    Args:
        x_eval: Tensor of shape [batch, in_dim] (e.g. [4, 3])
        y_eval: Tensor of shape [batch, in_dim, out_dim]
        grid:   Tensor of shape [in_dim, num_grid_points]
        k:      Spline order
        lamb:   Regularization parameter

    Returns:
        coef: Tensor of shape [in_dim, out_dim, n_coef]
    '''
    # Here, we assume the evaluation (sample) axis is the batch dimension.
    # x_eval: [batch, in_dim]
    # y_eval: [batch, in_dim, out_dim]
    batch, in_dim, out_dim = x_eval.shape[0], x_eval.shape[1], y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1

    # Evaluate B-spline basis matrix
    mat = B_batch(x_eval, grid, k)  # Expected shape: [batch, something, n_coef]
    # Rearrange mat to shape [in_dim, out_dim, batch, n_coef]
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)
    
    # Rearrange y_eval from [batch, in_dim, out_dim] to [in_dim, out_dim, batch, 1]
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(-1)
    
    device = mat.device
    # Compute least-squares terms using einsum.
    XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0, 1, 3, 2), mat)
    Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0, 1, 3, 2), y_eval)
    
    identity = torch.eye(n_coef, n_coef, device=device)[None, None, :, :].expand(in_dim, out_dim, n_coef, n_coef)
    A = XtX + lamb * identity
    B = Xty
    coef = (A.pinverse() @ B)[:, :, :, 0]
    
    # Compute endpoint slopes using the first two and last two samples along the batch axis.
    # x_eval: [batch, in_dim] -> use x_eval[0] and x_eval[1] for the start, and x_eval[-2], x_eval[-1] for the end.
    # Compute differences per in_dim and unsqueeze to shape [in_dim, 1]
    delta_x_start = (x_eval[1, :] - x_eval[0, :]).unsqueeze(1)
    delta_x_end   = (x_eval[-1, :] - x_eval[-2, :]).unsqueeze(1)
    
    # y_eval is currently [in_dim, out_dim, batch, 1].
    slope_start = (y_eval[:, :, 1, 0] - y_eval[:, :, 0, 0]) / delta_x_start
    slope_end   = (y_eval[:, :, -1, 0] - y_eval[:, :, -2, 0]) / delta_x_end
    
    # Adjust coefficients so that the endpoint slopes are preserved:
    coef[:, :, 1] = coef[:, :, 0] + (grid[:, k+1] - grid[:, k])[:, None] * slope_start
    coef[:, :, -1] = coef[:, :, -2] + (grid[:, -k-1] - grid[:, -k-2])[:, None] * slope_end
    
    return coef



def extend_grid(grid, k_extend=0):
    '''
    extend grid
    '''
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid