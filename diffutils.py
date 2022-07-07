
import tensorflow as tf

def D1(x_t, t, n):
    """Differentiate x_t with respect to t n times

    Args:
        x_t (function): any function that accepts tf tensor argument
        t (tf tensor): variable with which derivative will be calculated
        n (integer): number of times to differentiate

    Returns:
        list: list containing gradient
    """
    if n==0:
        return x_t(t)
    elif n>=1:
        with tf.GradientTape() as tape:
            tape.watch(t)
            dn_minus_1_xdt = D1(x_t, t, n-1)
        grad = tape.gradient(dn_minus_1_xdt, t, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return grad
    else:
        print(f'Invalid n: {n}')

def D2(f_xy, x, y, nx, ny):
    """Differentiate f_xy with respect to x (y) nx (ny) times, respectively

    Args:
        f_xy (function): function accepting two tensorflow tensors
        x (tf tensor): variable
        y (tf tensor): variable
        nx (integer): number of times f_xy is to be differentiated w.r.t. x
        ny (integer): number of times f_xy is to be differentiated w.r.t. y

    Returns:
        _type_: _description_
    """
    if ny==0 and nx==0:
        xy = tf.concat([x, y], axis=1)
        return f_xy(xy)
    elif ny==0 and nx>=1:
        with tf.GradientTape() as xtape:
            xtape.watch(x)
            res = D2(f_xy, x, y, nx-1, ny)
        grad = xtape.gradient(res, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return grad
    elif ny>=1 and nx>=0:
        with tf.GradientTape() as ytape:
            ytape.watch(y)
            res = D2(f_xy, x, y, nx, ny-1)
        grad = ytape.gradient(res, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return grad
    else:
        print(f'Invalid n: {nx} {ny}')

def Laplacian(f_xy, x, y):
    """Laplacian of f

    Args:
        f_xy (function): a function accepting two tf tensors
        x (tf tensor): independent variable
        y (tf tensor): independent variable

    Returns:
        list: list containing laplacian of function f_xy
    """
    x_tens = tf.convert_to_tensor(x)
    y_tens = tf.convert_to_tensor(y)
    return D2(f_xy, x_tens, y_tens, 2, 0) + D2(f_xy, x_tens, y_tens, 0, 2)