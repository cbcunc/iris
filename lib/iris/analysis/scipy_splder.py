import numpy as np

# A copy of the splder source code from scipy version 0.13.0, 
# which hasn't been released yet.


def splder(tck, n=1):
    """
    Compute the spline representation of the derivative of a given spline

    .. versionadded:: 0.13.0

    Parameters
    ----------
    tck : tuple of (t, c, k)
    Spline whose derivative to compute
    n : int, optional
    Order of derivative to evaluate. Default: 1

    Returns
    -------
    tck_der : tuple of (t2, c2, k2)
    Spline of order k2=k-n representing the derivative
    of the input spline.

    See Also
    --------
    splantider, splev, spalde

    Examples
    --------
    This can be used for finding maxima of a curve:

    >>> from scipy.interpolate import splrep, splder, sproot
    >>> x = np.linspace(0, 10, 70)
    >>> y = np.sin(x)
    >>> spl = splrep(x, y, k=4)

    Now, differentiate the spline and find the zeros of the
    derivative. (NB: `sproot` only works for order 3 splines, so we
    fit an order 4 spline):

    >>> dspl = splder(spl)
    >>> sproot(dspl) / np.pi
    array([ 0.50000001, 1.5 , 2.49999998])

    This agrees well with roots :math:`\pi/2 + n\pi` of
    :math:`\cos(x) = \sin'(x)`.

    """
    if n < 0:
        return splantider(tck, -n)

    t, c, k = tck

    if n > k:
        raise ValueError(("Order of derivative (n = %r) must be <= "
                          "order of spline (k = %r)") % (n, tck[2]))

    with np.errstate(invalid='raise', divide='raise'):
        try:
            for j in range(n):
                # See e.g. Schumaker, Spline Functions:Basic Theory, Chapter 5

                # Compute the denominator in the differentiation formula.
                dt = t[k+1:-1] - t[1:-k-1]
                # Compute the new coefficients
                c = (c[1:-1-k] - c[:-2-k]) * k / dt
                # Pad coefficient array to same size as knots
                # (FITPACK convention)
                c = np.r_[c, [0]*k]
                # Adjust knots
                t = t[1:-1]
                k -= 1
        except FloatingPointError:
            raise ValueError(("The spline has internal repeated knots "
                              "and is not differentiable %d times") % n)

    return t, c, k


def splantider(tck, n=1):
    """
    Compute the spline for the antiderivative (integral) of a given spline.

    .. versionadded:: 0.13.0

    Parameters
    ----------
    tck : tuple of (t, c, k)
    Spline whose antiderivative to compute
    n : int, optional
    Order of antiderivative to evaluate. Default: 1

    Returns
    -------
    tck_ader : tuple of (t2, c2, k2)
    Spline of order k2=k+n representing the antiderivative of the input
    spline.

    See Also
    --------
    splder, splev, spalde

    Notes
    -----
    The `splder` function is the inverse operation of this function.
    Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo
    rounding error.

    Examples
    --------
    >>> from scipy.interpolate import splrep, splder, splantider, splev
    >>> x = np.linspace(0, np.pi/2, 70)
    >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
    >>> spl = splrep(x, y)

    The derivative is the inverse operation of the antiderivative,
    although some floating point error accumulates:

    >>> splev(1.7, spl), splev(1.7, splder(splantider(spl)))
    (array(2.1565429877197317), array(2.1565429877201865))

    Antiderivative can be used to evaluate definite integrals:

    >>> ispl = splantider(spl)
    >>> splev(np.pi/2, ispl) - splev(0, ispl)
    2.2572053588768486

    This is indeed an approximation to the complete elliptic integral
    :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:

    >>> from scipy.special import ellipk
    >>> ellipk(0.8)
    2.2572053268208538

    """
    if n < 0:
        return splder(tck, -n)

    t, c, k = tck

    for j in range(n):
        # This is the inverse set of operations to splder.

        # Compute the multiplier in the antiderivative formula.
        dt = t[k+1:] - t[:-k-1]
        # Compute the new coefficients
        c = np.cumsum(c[:-k-1] * dt) / (k + 1)
        c = np.r_[0, c, [c[-1]]*(k+2)]
        # New knots
        t = np.r_[t[0], t, t[-1]]
        k += 1

    return t, c, k
