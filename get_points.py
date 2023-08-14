import torch


def get_boundary_points(*args, n_points=15, device=torch.device("cpu"), requires_grad=True, rand=False):
    if len(args) == 1:
        t_domain = args[0]
        if rand:
            """
            Idea, not tested
            """
            t_space = torch.rand(n_points) * (t_domain[1] - t_domain[0]) + t_domain[0]
        else:
            t_space = torch.linspace(t_domain[0], t_domain[1], n_points)
        t_grid = t_space.reshape(-1, 1).to(device)
        t_grid.requires_grad = requires_grad
        return (t_grid, )
    elif len(args) == 2:
        x_domain = args[0]
        t_domain = args[1]
        if rand:
            """
            Idea, not tested
            """
            x_space = torch.rand(n_points) * (x_domain[1] - x_domain[0]) + x_domain[0]
            t_space = torch.rand(n_points) * (t_domain[1] - t_domain[0]) + t_domain[0]
        else:
            x_space = torch.linspace(x_domain[0], x_domain[1], n_points)
            t_space = torch.linspace(t_domain[0], t_domain[1], n_points)
        x_grid = x_space.reshape(-1, 1).to(device)
        t_grid = t_space.reshape(-1, 1).to(device)
        x_grid.requires_grad = requires_grad
        t_grid.requires_grad = requires_grad
        x0 = torch.full_like(t_grid, x_domain[0], requires_grad=requires_grad)
        x1 = torch.full_like(t_grid, x_domain[1], requires_grad=requires_grad)
        down = (x0, t_grid)
        up = (x1, t_grid)
        return down, up
    elif len(args) == 3:
        x_domain = args[0]
        y_domain = args[1]
        t_domain = args[2]
        if rand:
            """
            Idea, not tested
            """
            x_space = torch.rand(n_points) * (x_domain[1] - x_domain[0]) + x_domain[0]
            y_space = torch.rand(n_points) * (y_domain[1] - y_domain[0]) + y_domain[0]
            t_space = torch.rand(n_points) * (t_domain[1] - t_domain[0]) + t_domain[0]
        else:
            x_space = torch.linspace(x_domain[0], x_domain[1], n_points)
            y_space = torch.linspace(y_domain[0], y_domain[1], n_points)
            t_space = torch.linspace(t_domain[0], t_domain[1], n_points)
        x_grid, t_grid = torch.meshgrid(x_space, t_space, indexing="ij")
        y_grid, t_grid = torch.meshgrid(y_space, t_space, indexing="ij")
        x_grid = x_grid.reshape(-1, 1).to(device)
        y_grid = y_grid.reshape(-1, 1).to(device)
        t_grid = t_grid.reshape(-1, 1).to(device)
        x_grid.requires_grad = requires_grad
        y_grid.requires_grad = requires_grad
        t_grid.requires_grad = requires_grad
        x0 = torch.full_like(t_grid, x_domain[0], requires_grad=requires_grad)
        x1 = torch.full_like(t_grid, x_domain[1], requires_grad=requires_grad)
        y0 = torch.full_like(t_grid, y_domain[0], requires_grad=requires_grad)
        y1 = torch.full_like(t_grid, y_domain[1], requires_grad=requires_grad)
        down = (x_grid, y0, t_grid)
        up = (x_grid, y1, t_grid)
        left = (x0, y_grid, t_grid)
        right = (x1, y_grid, t_grid)
        return down, up, left, right
    else:
        raise Exception(f"Too many arguments: {len(args)}, expected 1, 2 or 3.")


def get_initial_points(*args, n_points=15, device=torch.device("cpu"), requires_grad=True, rand=False):
    if len(args) == 1:
        t_domain = args[0]
        if rand:
            """
            Idea, not tested
            """
            t0 = torch.rand(1) * (t_domain[1] - t_domain[0]) + t_domain[0]
        else:
            t0 = torch.linspace(t_domain[0], t_domain[0], 1)
        t0 = t0.reshape(-1, 1).to(device)
        t0.requires_grad = requires_grad
        return t0
    elif len(args) == 2:
        x_domain = args[0]
        t_domain = args[1]
        if rand:
            x_space = torch.rand(n_points) * (x_domain[1] - x_domain[0]) + x_domain[0]
        else:
            x_space = torch.linspace(x_domain[0], x_domain[1], n_points)
        x_grid = x_space.reshape(-1, 1).to(device)
        x_grid.requires_grad = requires_grad
        t0 = torch.full_like(x_grid, t_domain[0], requires_grad=requires_grad)
        return x_grid, t0
    elif len(args) == 3:
        x_domain = args[0]
        y_domain = args[1]
        t_domain = args[2]
        if rand:
            """
            Idea, not tested
            """
            x_space = torch.rand(n_points) * (x_domain[1] - x_domain[0]) + x_domain[0]
            y_space = torch.rand(n_points) * (y_domain[1] - y_domain[0]) + y_domain[0]
        else:
            x_space = torch.linspace(x_domain[0], x_domain[1], n_points)
            y_space = torch.linspace(y_domain[0], y_domain[1], n_points)
        x_grid, y_grid = torch.meshgrid(x_space, y_space, indexing="ij")
        x_grid = x_grid.reshape(-1, 1).to(device)
        y_grid = y_grid.reshape(-1, 1).to(device)
        x_grid.requires_grad = requires_grad
        y_grid.requires_grad = requires_grad
        t0 = torch.full_like(x_grid, t_domain[0], requires_grad=requires_grad)
        return x_grid, y_grid, t0
    else:
        raise Exception(f"Too many arguments: {len(args)}, expected 1, 2 or 3.")


def get_interior_points(*args, n_points=15, device=torch.device("cpu"), requires_grad=True, rand=False):
    if len(args) == 1:
        t_domain = args[0]
        if rand:
            """
            Idea, not tested
            """
            t_raw = torch.rand(n_points, requires_grad=requires_grad) * (t_domain[1] - t_domain[0]) + t_domain[0]
        else:
            t_raw = torch.linspace(t_domain[0], t_domain[1], n_points, requires_grad=requires_grad)
        t = t_raw.reshape(-1, 1).to(device)
        return t
    elif len(args) == 2:
        x_domain = args[0]
        t_domain = args[1]
        if rand:
            """
            Idea, not tested
            """
            x_raw = torch.rand(n_points, requires_grad=requires_grad) * (x_domain[1] - x_domain[0]) + x_domain[0]
            t_raw = torch.rand(n_points, requires_grad=requires_grad) * (t_domain[1] - t_domain[0]) + t_domain[0]
        else:
            x_raw = torch.linspace(x_domain[0], x_domain[1], n_points, requires_grad=requires_grad)
            t_raw = torch.linspace(t_domain[0], t_domain[1], n_points, requires_grad=requires_grad)
        grids = torch.meshgrid(x_raw, t_raw, indexing="ij")
        x = grids[0].reshape(-1, 1).to(device)
        t = grids[1].reshape(-1, 1).to(device)
        return x, t
    elif len(args) == 3:
        x_domain = args[0]
        y_domain = args[1]
        t_domain = args[2]
        if rand:
            """
            Idea, not tested
            """
            x_raw = torch.rand(n_points, requires_grad=requires_grad) * (x_domain[1] - x_domain[0]) + x_domain[0]
            y_raw = torch.rand(n_points, requires_grad=requires_grad) * (y_domain[1] - y_domain[0]) + y_domain[0]
            t_raw = torch.rand(n_points, requires_grad=requires_grad) * (t_domain[1] - t_domain[0]) + t_domain[0]
        else:
            x_raw = torch.linspace(x_domain[0], x_domain[1], n_points, requires_grad=requires_grad)
            y_raw = torch.linspace(y_domain[0], y_domain[1], n_points, requires_grad=requires_grad)
            t_raw = torch.linspace(t_domain[0], t_domain[1], n_points, requires_grad=requires_grad)
        grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")
        x = grids[0].reshape(-1, 1).to(device)
        y = grids[1].reshape(-1, 1).to(device)
        t = grids[2].reshape(-1, 1).to(device)
        return x, y, t
    else:
        raise Exception(f"Too many arguments: {len(args)}, expected 1, 2 or 3.")

