module mir.optim.internal;

import mir.ndslice;

import mir.blas;
import mir.utility: min;

alias T = double;

enum mallocErrorMsg = "Failed to allocate memory.";
version(D_Exceptions)
    static immutable mallocError = new Error(mallocErrorMsg);

auto safeAlloc(T)(size_t n) @trusted
{
    import mir.internal.memory;
    auto workPtr = cast(T*)malloc(T.sizeof * n);
    if (workPtr is null)
    {
        version(D_Exceptions)
            throw mallocError;
        else
            assert(0, msg);
    }
    return workPtr[0 .. n];
}

@safe pure nothrow @nogc
void divBy(Slice!(T*) target, Slice!(const(T)*) roots)
{
    foreach (i; 0 .. target.length)
        target[i] *= 1 / roots[i];
}

@safe pure nothrow @nogc
void mulBy(Slice!(T*) target, Slice!(const(T)*) roots)
{
    foreach (i; 0 .. target.length)
        target[i] *= roots[i];
}

@safe pure nothrow @nogc
void eigenTimes(
    Slice!(const(T)*, 2, Canonical) eigenVectors,
    Slice!(const(T)*) eigenValues,
    Slice!(const(T)*) x,
    Slice!(T*) temp,
    Slice!(T*) y,
    T alpha = 1,
)
{
    svdTimes(eigenVectors, eigenVectors, eigenValues, x, temp, y, alpha);
}

@safe pure nothrow @nogc
void eigenSolve(
    Slice!(const(T)*, 2, Canonical) eigenVectors,
    Slice!(const(T)*) eigenValues,
    Slice!(const(T)*) y,
    Slice!(T*) temp,
    Slice!(T*) x,
    T alpha = 1,
)
{
    svdSolve(eigenVectors, eigenVectors, eigenValues, y, temp, x, alpha);
}

@safe pure nothrow @nogc
void eigenSqrtTimes(
    Slice!(const(T)*, 2, Canonical) eigenVectors,
    Slice!(const(T)*) eigenValuesRoots,
    Slice!(const(T)*) x,
    Slice!(T*) y,
)
in {
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenValuesRoots.length <= eigenVectors.length);
    assert(x.length == eigenVectors.length);
    assert(y.length == eigenValuesRoots.length);
}
do {
    auto r = eigenValuesRoots.length;
    gemv!T(1, eigenVectors[0 .. r], x, 0, y);
    y.mulBy(eigenValuesRoots);
}

@safe pure nothrow @nogc
void eigenSqrtSolve(
    Slice!(const(T)*, 2, Canonical) eigenVectors,
    Slice!(const(T)*) eigenValuesRoots,
    Slice!(T*) y,
    Slice!(T*) x,
)
in {
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenValuesRoots.length <= eigenVectors.length);
    assert(x.length == eigenVectors.length);
    assert(y.length == eigenValuesRoots.length);
}
do {
    auto r = eigenValuesRoots.length;
    y.divBy(eigenValuesRoots);
    gemv!T(1, eigenVectors[0 .. r].transposed, y, 0, x);
}

@safe pure nothrow @nogc
void eigenSqrtSplit(
    Slice!(const(T)*, 2, Canonical) eigenVectors,
    Slice!(const(T)*) eigenValuesRoots,
    Slice!(const(T)*) x,
    Slice!(T*) y,
)
in {
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenValuesRoots.length <= eigenVectors.length);
    assert(x.length == eigenVectors.length);
    assert(y.length == eigenValuesRoots.length);
}
do {
    auto r = eigenValuesRoots.length;
    gemv!T(1, eigenVectors[0 .. r], x, 0, y);
    y.divBy(eigenValuesRoots);
}

@safe pure nothrow @nogc
void eigenSqrtSplitReverse(
    Slice!(const(T)*, 2, Canonical) eigenVectors,
    Slice!(const(T)*) eigenValuesRoots,
    Slice!(T*) y,
    Slice!(T*) x,
)
in {
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenValuesRoots.length <= eigenVectors.length);
    assert(x.length == eigenVectors.length);
    assert(y.length == eigenValuesRoots.length);
}
do {
    auto r = eigenValuesRoots.length;
    y.mulBy(eigenValuesRoots);
    gemv!T(1, eigenVectors.transposed[0 .. r], y, 0, x);
}

@safe pure nothrow @nogc
void svdTimes(
    Slice!(const(T)*, 2, Canonical) leftSingularVectors,
    Slice!(const(T)*, 2, Canonical) rightSingularVectors,
    Slice!(const(T)*) singularValues,
    Slice!(const(T)*) x,
    Slice!(T*) temp,
    Slice!(T*) y,
    T alpha = 1,
)
in {
    assert(leftSingularVectors.length!0 == leftSingularVectors.length!1);
    assert(rightSingularVectors.length!0 == rightSingularVectors.length!1);
    assert(singularValues.length <= min(leftSingularVectors.length, rightSingularVectors.length));
    assert(singularValues.length <= temp.length);
    assert(x.length == rightSingularVectors.length);
    assert(y.length == leftSingularVectors.length);
}
do {
    auto r = singularValues.length;
    temp = temp[0 ..r];
    gemv!T(1, rightSingularVectors[0 .. r], x, 0, temp);
    temp.mulBy(singularValues);
    gemv!T(alpha, leftSingularVectors[0 .. r].transposed, temp, 0, y);
}

@safe pure nothrow @nogc
void svdSolve(
    Slice!(const(T)*, 2, Canonical) leftSingularVectors,
    Slice!(const(T)*, 2, Canonical) rightSingularVectors,
    Slice!(const(T)*) singularValues,
    Slice!(const(T)*) y,
    Slice!(T*) temp,
    Slice!(T*) x,
    T alpha = 1,
)
in {
    assert(leftSingularVectors.length!0 == leftSingularVectors.length!1);
    assert(rightSingularVectors.length!0 == rightSingularVectors.length!1);
    assert(singularValues.length <= min(leftSingularVectors.length, rightSingularVectors.length));
    assert(singularValues.length <= temp.length);
    assert(x.length == rightSingularVectors.length);
    assert(y.length == leftSingularVectors.length);
}
do {
    auto r = singularValues.length;
    temp = temp[0 ..r];
    gemv!T(1, leftSingularVectors[0 .. $, 0 .. r].transposed, y, 0, temp);
    temp.divBy(singularValues);
    gemv!T(alpha, rightSingularVectors[0 .. $, 0 .. r], temp, 0, x);
}

version(none):

///
size_t sytrf_rk(T)(
    char uplo,
    Slice!(T*, 2, Canonical) a,
    Slice!(T*) e,
    Slice!(lapackint*) ipiv,
    Slice!(T*) work,
    )
in
{
    assert(a.length!0 == a.length!1, "sytrf_rk: 'a' must be a square matrix.");
    assert(e.length == a.length, "sytrf_rk: 'e' must have the same length as 'a'.");
}
do
{
    lapackint info = void;
    lapackint n = cast(lapackint) a.length;
    lapackint lda = cast(lapackint) a._stride.max(1);
    lapackint lwork = cast(lapackint) work.length;
    lapack.sytrf_rk_(uplo, n, a.iterator, lda, e.iterator, ipiv.iterator, work.iterator, lwork, info);
    assert(info >= 0);
    return info;
}

unittest
{
    alias s = sytrf_rk!float;
    alias d = sytrf_rk!double;
    alias c = sytrf_rk!cfloat;
    alias z = sytrf_rk!cdouble;
}


///
size_t sytrf_rk_wk(T)(
    char uplo,
    Slice!(T*, 2, Canonical) a,
    )
in
{
    assert(a.length!0 == a.length!1, "sytrf_rk_wk: 'a' must be a square matrix.");
}
do
{

    lapackint info = void;
    lapackint n = cast(lapackint) a.length;
    lapackint lda = cast(lapackint) a._stride.max(1);
    lapackint lwork = -1;
    lapackint info = void;
    T e = void;
    T work = void;
    lapackint ipiv = void;

    lapack.sytrf_rk_(uplo, n, a.iterator, lda, &e, &ipiv, &work, lwork, info);

    return cast(size_t) work;
}

unittest
{
    alias s = sytrf_rk!float;
    alias d = sytrf_rk!double;
}

///
size_t sytrf_wk(T)(
    char uplo,
    Slice!(T*, 2, Canonical) a,
    )
in
{
    assert(a.length!0 == a.length!1, "sytrf_wk: 'a' must be a square matrix.");
}
do
{

    lapackint info = void;
    lapackint n = cast(lapackint) a.length;
    lapackint lda = cast(lapackint) a._stride.max(1);
    lapackint lwork = -1;
    lapackint info = void;
    T work = void;
    lapackint ipiv = void;

    lapack.sytrf_(uplo, n, a.iterator, lda, &ipiv, &work, lwork, info);

    return cast(size_t) work;
}

unittest
{
    alias s = sytrf_wk!float;
    alias d = sytrf_wk!double;
}

///
size_t sytrs_3(T)(
    char uplo,
    Slice!(T*, 2, Canonical) a,
    Slice!(T*) e,
    Slice!(lapackint*) ipiv,
    Slice!(T*, 2, Canonical) b,
    Slice!(T*) work,
    )
in
{
    assert(a.length!0 == a.length!1, "sytrs_3: 'a' must be a square matrix.");
    assert(e.length == a.length, "sytrs_3: 'e' must have the same length as 'a'.");
    assert(b.length!1 == a.length, "sytrs_3: 'b.length!1' must must be equal to 'a.length'.");
}
do
{
    lapackint n = cast(lapackint) a.length;
    lapackint nrhs = cast(lapackint) b.length;
    lapackint lda = cast(lapackint) a._stride.max(1);
    lapackint ldb = cast(lapackint) b._stride.max(1);
    lapackint info = void;
    lapack.sytrs_3_(uplo, n, nrhs, a.iterator, lda, e.iterator, ipiv.iterator, b.iterator, ldb, work.iterator, info);
    assert(info >= 0);
    return info;
}

unittest
{
    alias s = sytrs_3!float;
    alias d = sytrs_3!double;
    alias c = sytrs_3!cfloat;
    alias z = sytrs_3!cdouble;
}

///
auto iamax(T,
    SliceKind kindX,
    )(
    Slice!(const(T)*, 1, kindX) x,
    )
{
    return cblas.iamax(
        cast(cblas.blasint) x.length,
        x.iterator,
        cast(cblas.blasint) x._stride,
    );
}

auto famaxElem(T, SliceKind kind)(Slice!(const(T)*, 1, kind) x)
{
    return reduce!fmax(-T.infinity, x.map!fabs);
}

auto infNormDistance(T)(Slice!(const(T)*) x, Slice!(const(T)*) y)
{
    return reduce!fmax(-T.infinity, map!fabs(x - y));
}

auto faminElem(T, SliceKind kind)(Slice!(const(T)*, 1, kind) x)
{
    return reduce!fmin(+T.infinity, x.map!fabs);
}

struct QPSolver(T)
    if (is(T == float) || is(T == double))
{
    import mir.math.common: sqrt, fmin, fmax;
    import mir.math.constant: GoldenRatio;

    T sigma = T.epsilon.sqrt;
    T rho = 1;
    T alpha = GoldenRatio;
    T epsRel = T.epsilon.sqrt.sqrt * 3;
    T epsAbs = T.epsilon.sqrt.sqrt * 3;
    T epsPrimalInfeasibility = T.epsilon.sqrt.sqrt;
    T epsDualInfeasibility = T.epsilon.sqrt.sqrt;

    size_t n;
    size_t m;

    Slice!(T, 1, Universal) sigmaI() @trusted @property
    {
        return typeof(return)(&sigma, [n], [sizediff_t(1)]);
    }

    Slice!(T*, 2) A;  // n x m
    Slice!(T*, 2) P; // P n
    Slice!(T*, 2) F; // LDL^t factorization of (P + σI + ρA^tA)

    Slice!(T*) q; // n
    Slice!(T*) l; // n
    Slice!(T*) u; // n

    Slice!(lapackint*) F_ipiv;
    Slice!(T*) F_work;
    Slice!(T*) F_e; // n

    Slice!(T*) x; // n
    Slice!(T*) x_prev; // n
    Slice!(T*, 4) buff_4n; // 2 * n

    Slice!(T*) y; // m
    Slice!(T*) y_prev; // m
    Slice!(T*) z; // m
    Slice!(T*) Ax; // m
    Slice!(T*) Ax_prev; // m
    Slice!(T*) buff_m; // m

    void updateF()
    {
        mir.blas.copy!T(P.flattened, F.flattened);
        mir.blas.axpy!T(1, sigmaI, F.diagonal);
        mir.blas.syrk!T(Uplo.lower, rho, A.transposed, 1, F);
        sytrf_rk!T('U', F, F_e, F_ipiv, F_work);
    }

    QPTerminationCriteria iterate()
    {
        mir.blas.copy!T(y, y_prev);
        mir.blas.scale!T(-1, y);
        mir.blas.axpy!T(rho, z, y);
        mir.blas.copy!T(x, x_prev);
        if (useA)
            mir.blas.gemv!T(1, A.transposed, y, 0, x);
        else
            mir.blas.copy!T(y, x);
        mir.blas.axpy!T(-1, q, x);
        mir.blas.axpy!T(sigma, x_prev, x);
        if (useA || useP)
            sytrs_3!T('U', F, F_e, F_ipiv, x(1, n));
        else
            mir.blas.scale(1 / (1 + sigma + rho), x);
        mir.blas.scale!T(alpha, x);
        mir.blas.scale!T(1 - alpha, z);
        mir.blas.copy!T(Ax, Ax_prev);
        mir.blas.scale!T(1 - alpha, Ax);
        if (useA)
            mir.blas.gemv!T(1, A, x, 0, y);
        else
            mir.blas.copy!T(x, y);
        mir.blas.axpy!T(1, y, z);
        mir.blas.axpy!T(1, y, Ax);
        mir.blas.axpy!T(1 - alpha, x_prev, x);
        mir.blas.copy!T(z, y);
        mir.blas.axpy!T(1 / rho, y_prev, z);
        foreach (i, ref e; z.field)
            e = e.fmax(l[i]).fmin(u[i]);
        mir.blas.axpy!T(-1, z, y);
        mir.blas.scale!T(rho, y);
        mir.blas.axpy!T(1, y_prev, y);

        mir.blas.copy!T(Ax, buff_4n[0]);
        mir.blas.axpy!T(-1, z, buff_4n[0]);
        if (famaxElem(buff_4n[0]) < epsAbs + epsRel * famaxElem(z).fmax(famaxElem(Ax)))
            return QPTerminationCriteria.primal;
        return QPTerminationCriteria.none;
    }

    QPTerminationCriteria criteria()
    {
        if (useA)
        {
            mir.blas.copy!T(x, buff_4n[1]);
            mir.blas.axpy!T(-1, x_prev, buff_4n[1]);
            if (useP)
                mir.blas.symm!T(Side.right, Uplo.lower, 1, P, buff_4n[0 .. 2], 0, buff_4n[2 .. 4]);
            else
                mir.blas.copy(buff_4n[0 .. 2].flattened, 0, buff_4n[2 .. 4].flattened);
            mir.blas.gemv!T(1, A.transposed, y, 0, buff_4n[0]);
        }
        else
        {
            if (useP)
                mir.blas.symv!T(Uplo.lower, 1, P, x, 0, buff_4n[2]);
            else
                mir.blas.copy(x, buff_4n[2]);
            mir.blas.copy!T(y, buff_4n[0]);
        }

        T prim = famaxElem(q)
            .fmax(famaxElem(buff_4n[2]))
            .fmax(famaxElem(buff_4n[0]));
        mir.blas.axpy(1, q, buff_4n[0]);
        mir.blas.axpy(1, buff_4n[2], buff_4n[0]);

        if (famaxElem(buff_4n[0]) < epsAbs + epsRel * prim)
            return QPTerminationCriteria.dual;

        if (useA)
        {
            T normPdx = famaxElem(buff_4n[2] - buff_4n[3]);
        }

        return QPTerminationCriteria.none;
    }


    // uses buff_n as new #x
    void update()
    {
    }
}
