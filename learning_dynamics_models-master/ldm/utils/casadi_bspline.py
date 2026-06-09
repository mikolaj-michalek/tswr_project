import casadi as ca

def build_casadi_bspline(knots, control_points, degree):
    """
    Builds an optimized CasADi function for a clamped B-spline.
    """
    # 1. Define the symbolic input
    x = ca.MX.sym('x')
    
    # 2. Format inputs correctly
    # The 'knots' must be a list of lists (one list per dimension)
    # Ensure they are standard Python floats, not DM objects.
    knots_list = [list(knots)]
    
    # Control points can stay as a DM or list
    coeffs = ca.DM(control_points)
    
    # 3. Create the B-spline expression
    # Prototype: bspline(x, coeffs, [knots], [degrees], m)
    # m is the dimensionality of the output (usually 1 for a single curve)
    spline_expr = ca.bspline(x, coeffs, knots_list, [degree], 1)
    
    # 4. Wrap into a Function
    opts = {
        "jit": True,
        "compiler": "shell",
        "jit_options": {"flags": ["-O3"]}
    }
    
    return ca.Function('bspline_eval', [x], [spline_expr], opts)

if __name__ == "__main__":
    # --- Example with correct types ---
    p = 3
    # Standard Python list of floats
    knot_vector = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0] 
    # 5 control points for this knot/degree combo
    ctrl_pts = [0.0, 1.0, -1.0, 2.0, 0.5]

    bspline_f = build_casadi_bspline(knot_vector, ctrl_pts, p)

    # Test it
    print(f"Result at 0.5: {bspline_f(0.5)}")