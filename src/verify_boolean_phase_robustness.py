#!/usr/bin/env python3
"""Exact verifier for the Z-preserving robustness certificate.

The script verifies the four scalar determinant polynomials that arise in
Lemma 2 of boolean_phase_robustness_proof.tex.  All calculations are exact in
Q(sqrt(2)); no floating-point decision is used.
"""
from collections import deque
from fractions import Fraction as F
from math import comb
import sympy as sp

q = sp.sqrt(2)
k = 1 + q
x, y, X, Y, U, V = sp.symbols("x y X Y U V")

sa = 2*x/(1+x*x)
ca = (1-x*x)/(1+x*x)
sb = 2*y/(1+y*y)
cb = (1-y*y)/(1+y*y)
e = sp.factor(2*sb*cb)            # sin(2b)
d = sp.factor(cb*cb-sb*sb)       # cos(2b)
p = sp.factor(k*(sa+ca-1))
h = sp.factor(k*(sb+cb-1))
gamma = 3*k

def determinant(u, v, w, r):
    return sp.factor(
        (gamma+r*w)**2
        - (u+r*v-gamma*(ca+r*sa*e))**2
        - (gamma*sa*d)**2
    )

den = (1+x*x)**2*(1+y*y)**3
D = {
    "lp": determinant(1, p*h, p*h, +1),
    "lm": determinant(1, p*h, p*h, -1),
    "hp": determinant(p, h, p*h, +1),
    "hm": determinant(p, h, p*h, -1),
}
t0 = q-1
N = {}
for name, expr in D.items():
    numerator = sp.expand(sp.cancel(expr*den))
    substitution = (
        {x: t0*X, y: t0*Y}
        if name.startswith("l")
        else {x: 1-(1-t0)*X, y: t0*Y}
    )
    N[name] = sp.Poly(
        sp.expand(numerator.subs(substitution)), X, Y, extension=q
    )


def normalized_polynomials_for_slope(slope):
    """Return the four cleared unit-square polynomials at an exact slope."""

    identity_coefficient = -3 + 8*q*slope
    deficit_coefficient = 8*slope

    def determinant_at_slope(u, v, w, r):
        return sp.factor(
            (identity_coefficient+r*w)**2
            - (
                u+r*v
                - deficit_coefficient*(ca+r*sa*e)
            )**2
            - (deficit_coefficient*sa*d)**2
        )

    determinants = {
        "lp": determinant_at_slope(1, p*h, p*h, +1),
        "lm": determinant_at_slope(1, p*h, p*h, -1),
        "hp": determinant_at_slope(p, h, p*h, +1),
        "hm": determinant_at_slope(p, h, p*h, -1),
    }
    polynomials = {}
    for name, expr in determinants.items():
        numerator = sp.expand(sp.cancel(expr*den))
        substitution = (
            {x: t0*X, y: t0*Y}
            if name.startswith("l")
            else {x: 1-(1-t0)*X, y: t0*Y}
        )
        polynomials[name] = sp.Poly(
            sp.expand(numerator.subs(substitution)),
            X,
            Y,
            extension=q,
        )
    return polynomials


def split_qsqrt2(expr):
    expr = sp.collect(sp.expand(expr), q)
    a = sp.simplify(expr.subs(q, 0))
    b = sp.simplify((expr-a)/q)
    return (
        F(int(sp.numer(a)), int(sp.denom(a))),
        F(int(sp.numer(b)), int(sp.denom(b))),
    )

def add(z, w):
    return (z[0]+w[0], z[1]+w[1])

def mul_rational(z, r):
    return (z[0]*r, z[1]*r)

def sign_qsqrt2(z):
    """Exact sign of a+b*sqrt(2), with rational a,b."""
    a, b = z
    if b == 0:
        return (a > 0) - (a < 0)
    if a == 0:
        return (b > 0) - (b < 0)
    if a > 0 and b > 0:
        return 1
    if a < 0 and b < 0:
        return -1
    if a > 0 and b < 0:
        return 1 if a*a > 2*b*b else (-1 if a*a < 2*b*b else 0)
    return 1 if 2*b*b > a*a else (-1 if 2*b*b < a*a else 0)

def abs_qsqrt2_expr(expr):
    return expr if sign_qsqrt2(split_qsqrt2(expr)) >= 0 else -expr

def power_coefficients(poly, nx=4, ny=6):
    return {
        (i, j): split_qsqrt2(poly.coeff_monomial(X**i*Y**j))
        for i in range(nx+1) for j in range(ny+1)
    }

def affine_power(coeff, lx, hx, ly, hy, nx=4, ny=6):
    out = {(i, j): (F(0), F(0)) for i in range(nx+1) for j in range(ny+1)}
    for i in range(nx+1):
        for j in range(ny+1):
            z = coeff[(i, j)]
            if z == (0, 0):
                continue
            for p0 in range(i+1):
                fx = F(comb(i, p0))*lx**(i-p0)*hx**p0
                for q0 in range(j+1):
                    fac = fx*F(comb(j, q0))*ly**(j-q0)*hy**q0
                    out[(p0, q0)] = add(out[(p0, q0)], mul_rational(z, fac))
    return out

def bernstein_coefficients(power, nx=4, ny=6):
    out = {}
    for i in range(nx+1):
        for j in range(ny+1):
            z = (F(0), F(0))
            for p0 in range(i+1):
                fx = F(comb(i, p0), comb(nx, p0))
                for q0 in range(j+1):
                    fac = fx*F(comb(j, q0), comb(ny, q0))
                    z = add(z, mul_rational(power[(p0, q0)], fac))
            out[(i, j)] = z
    return out

def verify_box(poly, box):
    lx, ux, ly, uy = box
    power = affine_power(power_coefficients(poly), lx, ux-lx, ly, uy-ly)
    bern = bernstein_coefficients(power)
    assert all(sign_qsqrt2(z) >= 0 for z in bern.values())


def verify_slope_one_single_box():
    """Verify the simpler slope-one theorem with four full-square boxes."""

    unit_square = (F(0), F(1), F(0), F(1))
    for polynomial in normalized_polynomials_for_slope(sp.Rational(1)).values():
        verify_box(polynomial, unit_square)
    return True


# Exact dyadic Bernstein cover of the complement of the local equality boxes.
delta = F(1, 64)
BOXES = {
    "lp": [
        (F(0), F(63,64), F(0), F(1)),
        (F(63,64), F(1), F(0), F(63,64)),
    ],
    "lm": [(F(0), F(1), F(0), F(1))],
    "hp": [
        (F(0),F(1,64),F(1,64),F(1)),
        (F(65,128),F(1),F(0),F(1,2)),
        (F(1,64),F(65,128),F(1,2),F(1)),
        (F(65,128),F(1),F(1,2),F(1)),
        (F(1,64),F(65,128),F(1,4),F(1,2)),
        (F(67,256),F(65,128),F(0),F(1,4)),
        (F(1,64),F(67,256),F(1,8),F(1,4)),
        (F(71,512),F(67,256),F(0),F(1,8)),
        (F(1,64),F(71,512),F(1,16),F(1,8)),
        (F(79,1024),F(71,512),F(0),F(1,16)),
        (F(1,64),F(79,1024),F(0),F(1,32)),
        (F(1,64),F(79,1024),F(1,32),F(1,16)),
    ],
    "hm": [
        (F(1,64),F(1),F(0),F(1)),
        (F(0),F(1,64),F(1,64),F(1)),
    ],
}


def verify_cover(boxes):
    """Check exactly that closed rational rectangles cover the unit square."""

    xs = sorted(
        {F(0), F(1)}
        | {box[0] for box in boxes}
        | {box[1] for box in boxes}
    )
    ys = sorted(
        {F(0), F(1)}
        | {box[2] for box in boxes}
        | {box[3] for box in boxes}
    )

    def covered(x0, y0):
        return any(
            lx <= x0 <= ux and ly <= y0 <= uy
            for lx, ux, ly, uy in boxes
        )

    probes = {(x0, y0) for x0 in xs for y0 in ys}
    probes.update(
        ((xs[i] + xs[i + 1]) / 2, y0)
        for i in range(len(xs) - 1)
        for y0 in ys
    )
    probes.update(
        (x0, (ys[j] + ys[j + 1]) / 2)
        for x0 in xs
        for j in range(len(ys) - 1)
    )
    probes.update(
        (
            (xs[i] + xs[i + 1]) / 2,
            (ys[j] + ys[j + 1]) / 2,
        )
        for i in range(len(xs) - 1)
        for j in range(len(ys) - 1)
    )
    return all(covered(x0, y0) for x0, y0 in probes)


def local_margin(name):
    if name == "lp":
        local = sp.Poly(
            sp.expand(N[name].as_expr().subs({X:1-U, Y:1-V})),
            U, V, extension=q,
        )
    else:
        local = sp.Poly(
            sp.expand(N[name].as_expr().subs({X:U, Y:V})),
            U, V, extension=q,
        )
    a = local.coeff_monomial(U**2)
    b = local.coeff_monomial(U*V)
    c = local.coeff_monomial(V**2)
    diagonal_min = (
        a
        if sign_qsqrt2(split_qsqrt2(sp.simplify(a - c))) <= 0
        else c
    )
    quadratic_lower = sp.simplify(diagonal_min-abs_qsqrt2_expr(b)/2)
    remainder = sp.simplify(sum(
        abs_qsqrt2_expr(coef)*sp.Rational(1,64)**(i+j-2)
        for (i,j), coef in local.terms() if i+j >= 3
    ))
    margin = sp.simplify(quadratic_lower-remainder)
    assert sign_qsqrt2(split_qsqrt2(margin)) > 0
    return margin

def main():
    local_boxes = {
        "lp": [(F(63,64), F(1), F(63,64), F(1))],
        "lm": [],
        "hp": [(F(0), F(1,64), F(0), F(1,64))],
        "hm": [(F(0), F(1,64), F(0), F(1,64))],
    }
    for name in ("lp", "lm", "hp", "hm"):
        assert verify_cover(BOXES[name] + local_boxes[name])
    for name, boxes in BOXES.items():
        for box in boxes:
            verify_box(N[name], box)
    margins = {name: local_margin(name) for name in ("lp", "hp", "hm")}
    print("Exact Bernstein certificates: PASS")
    print("Exact local remainder certificates: PASS")
    for name, margin in margins.items():
        print(f"  {name}: margin = {sp.factor(margin, extension=q)}")
    print("The four determinant polynomials are nonnegative on their full domains.")

if __name__ == "__main__":
    main()
