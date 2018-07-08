/++
$(H1 Nonlinear Least Squares Solver)

Copyright: Copyright © 2018, Symmetry Investments & Kaleidic Associates Advisory Limited
Authors:   Ilya Yaroshenko

Macros:
NDSLICE = $(REF_ALTTEXT $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
+/
module mir.least_squares;

import mir.ndslice.slice: Slice, SliceKind, Contiguous, ContiguousVector, ContiguousMatrix;
import std.meta;
import std.traits;

public import std.typecons: Flag, Yes, No;

version(D_BetterC)
{} else
{
    /++
    Exception for $(LREF optimize).
    +/
    class LeastSquaresLMException(T) : Exception
        if (is(T == double) || is(T == float))
    {
        ///
        LeastSquaresLM!T lm;

        ///
        @nogc @safe pure nothrow this(LMStatus st, ref LeastSquaresLM!T lm,  string file = __FILE__, size_t line = __LINE__, Throwable next = null)
        {
            this.lm = lm;
            super(st.lmStatusString, file, line, next);
        }

        ///
        @nogc @safe pure nothrow this(LMStatus st, ref LeastSquaresLM!T lm, Throwable next, string file = __FILE__, size_t line = __LINE__)
        {
            this.lm = lm;
            super(st.lmStatusString, file, line, next);
        }
    }
}

/++
Parameters:
    f = function
    g = Jacobian of `f`
    x = current (initial/final) `X` solution.
    y = current (final) `Y = f(X)`.
    tolX = search tolerance in x
    tolG = search tolerance in gradient
    maxIter = maximum number of iterations
    minStepQuality = for steps below this quality, the trust region is shrinked
    goodStepQuality = for steps above this quality, the trust region is expanded
    lambda = (inverse of) initial trust region radius
    lambdaIncrease = `lambda` is multiplied by this factor after step below min quality
    lambdaDecrease = `lambda` is multiplied by this factor after good quality steps
    lower = lower bound solution to these limits
    upper = upper bound solution to these limits
    maxLambda = minimum trust region radius
    minLambda = maximum trust region radius
    minDiagonal = lower bound on values of diagonal matrix used to regularize the trust region step
+/
struct LeastSquaresLM(T)
    if (is(T == double) || is(T == float))
{
    import lapack: lapackint;

    /// Default iteration parameters.
    enum T tolXDefault = T(2) ^^ ((1 - T.mant_dig) / 2);
    /// ditto
    enum T tolGDefault = T(2) ^^ ((1 - T.mant_dig) * 3 / 4);
    /// ditto
    enum T jacobianEpsilonDefault = T(2) ^^ ((1 - T.mant_dig) / 3);
    /// ditto
    enum T lambdaDefault = T(2) ^^ 3;
    /// ditto
    enum T lambdaIncreaseDefault = T(2) ^^ 3;
    /// ditto
    enum T lambdaDecreaseDefault = T(2) ^^ -3;
    /// ditto
    enum T minStepQualityDefault = T(2) ^^ -10;
    /// ditto
    enum T goodStepQualityDefault = 0.75;
    /// ditto
    enum T maxLambdaDefault = 1 / T.epsilon;
    /// ditto
    enum T minLambdaDefault = T.epsilon;
    /// ditto
    enum T minDiagonalDefault = T.epsilon;

    /// Delegates for low level D API.
    alias FunctionDelegate = void delegate(ContiguousVector!(const T) x, ContiguousVector!T y) @safe nothrow @nogc pure;
    /// ditto
    alias JacobianDelegate = void delegate(ContiguousVector!(const T) x, ContiguousMatrix!T J) @safe nothrow @nogc pure;

    /// Delegates for low level C API.
    alias FunctionFunction = extern(C) void function(void* context, size_t m, size_t n, const(T)* x, T* y) @system nothrow @nogc pure;
    ///
    alias JacobianFunction = extern(C) void function(void* context, size_t m, size_t n, const(T)* x, T* J) @system nothrow @nogc pure;

    private T* _lower_ptr;
    private T* _upper_ptr;
    private T* _x_ptr;
    private T* _deltaX_ptr;
    private T* _deltaXBase_ptr;
    private T* _mJy_ptr;
    private lapackint* _ipiv_ptr;
    private T* _y_ptr;
    private T* _mBuffer_ptr;
    private T* _JJ_ptr;
    private T* _J_ptr;
    private ContiguousVector!T _work;

    /++
    Y = f(X) dimension.
    
    Can be decresed after allocation to reuse existing data allocated in LM.
    +/
    size_t m;

    /++
    X dimension.

    Can be decresed after allocation to reuse existing data allocated in LM.
    +/
    size_t n;

    /++
    Iteration params.

    Default values can be set by $(LREF LeastSquaresLM) constructor,
        $(LREF LeastSquaresLM.reset), or $(LREF mir_least_squares_lm_reset).
    +/
    size_t maxIter;
    /// ditto
    T tolX = 0;
    /// ditto
    T tolG = 0;
    /// ditto
    T lambda = 0;
    /// ditto
    T lambdaIncrease = 0;
    /// ditto
    T lambdaDecrease = 0;
    /// ditto
    T minStepQuality = 0;
    /// ditto
    T goodStepQuality = 0;
    /// ditto
    T maxLambda = 0;
    /// ditto
    T minLambda = 0;
    /// ditto
    T minDiagonal = 0;
    /// ditto
    T jacobianEpsilon = 0;

    /++
    Counters and state values.
    +/
    size_t iterCt;
    /// ditto
    size_t fCalls;
    /// ditto
    size_t gCalls;
    /// ditto
    T residual = 0;
    /// ditto
    uint maxAge;
    /// ditto
    LMStatus status;
    /// ditto
    bool xConverged;
    /// ditto
    bool gConverged;

    /++
    Initialize iteration params and allocates vectors in GC and resets iteration counters, states a.
    Params:
        m = Y = f(X) dimension
        n = X dimension
        lowerBounds = flag to allocate lower bounds
        upperBounds = flag to allocate upper bounds
    +/
    this()(size_t m, size_t n, Flag!"lowerBounds" lowerBounds = No.lowerBounds, Flag!"upperBounds" upperBounds = No.upperBounds)
    {
        initParams;
        gcAlloc(m, n, lowerBounds, upperBounds);
    }

    /++
    Initialize default params and allocates vectors in GC.
    `lowerBounds` and `upperBounds` are binded to lm struct.
    +/
    this()(size_t m, size_t n, T[] lowerBounds, T[] upperBounds) @trusted
    {
        initParams;
        gcAlloc(m, n, false, false);
        if (lowerBounds)
        {
            assert(lowerBounds.length == n);
            _lower_ptr = lowerBounds.ptr;
        }
        if (upperBounds)
        {
            assert(upperBounds.length == n);
            _upper_ptr = upperBounds.ptr;
        }
    }

    @trusted pure nothrow @nogc @property
    {
        /++
        Returns: lower bounds if they were set or zero length vector otherwise.
        +/
        ContiguousVector!T lower() { return ContiguousVector!T([_lower_ptr ? n : 0], sizediff_t[0].init, _lower_ptr); }
        /++
        Returns: upper bounds if they were set or zero length vector otherwise.
        +/
        ContiguousVector!T upper() { return ContiguousVector!T([_upper_ptr ? n : 0], sizediff_t[0].init, _upper_ptr); }
        /++
        Returns: Current X vector.
        +/
        ContiguousVector!T x() { return ContiguousVector!T([n], sizediff_t[0].init, _x_ptr); }
        /++
        Returns: The last success ΔX.
        +/
        ContiguousVector!T deltaX() { return ContiguousVector!T([n], sizediff_t[0].init, _deltaX_ptr); }
        /++
        Returns: Current Y = f(X).
        +/
        ContiguousVector!T y() { return ContiguousVector!T([m], sizediff_t[0].init, _y_ptr); }
    private:
        ContiguousVector!T mJy() { return ContiguousVector!T([n], sizediff_t[0].init, _mJy_ptr); }
        ContiguousVector!T deltaXBase() { return ContiguousVector!T([n], sizediff_t[0].init, _deltaXBase_ptr); }
        ContiguousVector!lapackint ipiv() { return ContiguousVector!lapackint([n], sizediff_t[0].init, _ipiv_ptr); }
        ContiguousVector!T mBuffer() { return ContiguousVector!T([m], sizediff_t[0].init, _mBuffer_ptr); }
        ContiguousMatrix!T JJ() { return ContiguousMatrix!T([n, n], sizediff_t[0].init, _JJ_ptr); }
        ContiguousMatrix!T J() { return ContiguousMatrix!T([m, n], sizediff_t[0].init, _J_ptr); }
    }

    /++
    Resets all counters and flags, fills `x`, `y`, `upper`, `lower`, vecors with default values.
    +/
    pragma(inline, false)
    void reset()() @safe pure nothrow @nogc
    {
        status = LMStatus.initialized;
        maxAge = 0;
        iterCt = 0;
        fCalls = 0;
        gCalls = 0;     
        xConverged = false;
        gConverged = false;    
        fill(T.nan, x);
        fill(T.nan, y);
        fill(-T.infinity, lower);
        fill(+T.infinity, upper);
    }

    /++
    Initialize LM data structure with default params for iteration.
    +/
    pragma(inline, false)
    void initParams()() @safe pure nothrow @nogc
    {
        maxIter = 100;
        tolX = tolXDefault;
        tolG = tolGDefault;
        lambda = lambdaDefault;
        lambdaIncrease = lambdaIncreaseDefault;
        lambdaDecrease = lambdaDecreaseDefault;
        minStepQuality = minStepQualityDefault;
        goodStepQuality = goodStepQualityDefault;
        maxLambda = maxLambdaDefault;
        minLambda = minLambdaDefault;
        minDiagonal = minDiagonalDefault;
        jacobianEpsilon = jacobianEpsilonDefault;
    }

    /++
    Allocates data in GC.
    +/
    pragma(inline, false)
    auto gcAlloc()(size_t m, size_t n, bool lowerBounds = false, bool upperBounds = false) //nothrow @trusted pure
    {
        import mir.lapack: sysv_rook_wk;
        import mir.ndslice.allocation: uninitSlice;
        import mir.ndslice.slice: sliced;
        import mir.ndslice.topology: canonical;

        this.m = m;
        this.n = n;
        _lower_ptr = lowerBounds ? n.uninitSlice!T._iterator : null;
        _upper_ptr = upperBounds ? n.uninitSlice!T._iterator : null;
        _x_ptr = n.uninitSlice!T._iterator;
        _deltaX_ptr = n.uninitSlice!T._iterator;
        _mJy_ptr = n.uninitSlice!T._iterator;
        _deltaXBase_ptr = n.uninitSlice!T._iterator;
        _ipiv_ptr = n.uninitSlice!lapackint._iterator;
        _y_ptr = m.uninitSlice!T._iterator;
        _mBuffer_ptr = m.uninitSlice!T._iterator;
        _JJ_ptr = [n, n].uninitSlice!T._iterator;
        _J_ptr = [m, n].uninitSlice!T._iterator;
        _work = sysv_rook_wk('L', JJ.canonical, _deltaX_ptr.sliced([1, n]).canonical).uninitSlice!T;
        reset;
    }

    /++
    Allocates data using C Runtime.
    +/
    pragma(inline, false)
    void stdcAlloc()(size_t m, size_t n, bool lowerBounds = false, bool upperBounds = false) nothrow @nogc @trusted
    {
        import mir.lapack: sysv_rook_wk;
        import mir.ndslice.allocation: stdcUninitSlice;
        import mir.ndslice.slice: sliced;
        import mir.ndslice.topology: canonical;

        this.m = m;
        this.n = n;
        _lower_ptr = lowerBounds ? n.stdcUninitSlice!T._iterator : null;
        _upper_ptr = upperBounds ? n.stdcUninitSlice!T._iterator : null;
        _x_ptr = n.stdcUninitSlice!T._iterator;
        _deltaX_ptr = n.stdcUninitSlice!T._iterator;
        _mJy_ptr = n.stdcUninitSlice!T._iterator;
        _deltaXBase_ptr = n.stdcUninitSlice!T._iterator;
        _ipiv_ptr = n.stdcUninitSlice!lapackint._iterator;
        _y_ptr = m.stdcUninitSlice!T._iterator;
        _mBuffer_ptr = m.stdcUninitSlice!T._iterator;
        _JJ_ptr = [n, n].stdcUninitSlice!T._iterator;
        _J_ptr = [m, n].stdcUninitSlice!T._iterator;
        _work = sysv_rook_wk('L', JJ.canonical, _deltaX_ptr.sliced([1, n]).canonical).stdcUninitSlice!T;
        reset;
    }

    /++
    Frees vectors including `x`, `y`, `upper`, `lower`. Use in pair with `.stdcAlloc`.
    +/
    pragma(inline, false)
    void stdcFree()() nothrow @nogc @trusted
    {
        import core.stdc.stdlib: free;
        if (_lower_ptr) _lower_ptr.free;
        if (_upper_ptr) _upper_ptr.free;
        _x_ptr.free;
        _deltaX_ptr.free;
        _mJy_ptr.free;
        _deltaXBase_ptr.free;
        _ipiv_ptr.free;
        _y_ptr.free;
        _mBuffer_ptr.free;
        _JJ_ptr.free;
        _J_ptr.free;
        _work._iterator.free;
    }

    // size_t toHash() @safe pure nothrow @nogc
    // {
    //     return size_t(0);
    // }
    // size_t __xtoHash() @safe pure nothrow @nogc
    // {
    //     return size_t(0);
    // }
}

/++
High level D API for Levenberg-Marquardt Algorithm.

Computes the argmin over x of `sum_i(f(x_i)^2)` using the Modified Levenberg-Marquardt
algorithm, and an estimate of the Jacobian of `f` at x.

The function `f` should take an input vector of length `n`, and fill an output
vector of length `m`.

The function `g` is the Jacobian of `f`, and should fill a row-major `m x n` matrix. 

Throws: $(LREF LeastSquaresLMException)
Params:
    f = `n -> m` function
    g = `m × n` Jacobian (optional)
    tm = thread manager for finite difference jacobian approximation in case of g is null (optional)
    lm = Levenberg-Marquardt data structure
    taskPool = task Pool with `.parallel` method for finite difference jacobian approximation in case of g is null (optional)
See_also: $(LREF optimizeImpl)
+/
void optimize(alias f, alias g = null, alias tm = null, T)(scope ref LeastSquaresLM!T lm)
    if ((is(T == float) || is(T == double)) && __traits(compiles, optimizeImpl!(f, g, tm, T)))
{
    if (auto err = optimizeImpl!(f, g, tm, T)(lm))
        throw new LeastSquaresLMException!T(err, lm);
}

/// ditto
void optimize(alias f, TaskPool, T)(scope ref LeastSquaresLM!T lm, TaskPool taskPool)
    if (is(T == float) || is(T == double))
{
    auto tm = delegate(size_t count, void* taskContext, scope LeastSquaresTask task)
    {
        version(all)
        {
            import mir.ndslice.topology: iota;
            foreach(i; taskPool.parallel(count.iota))
                task(taskContext, taskPool.size, taskPool.workerIndex - 1, i);
        }
        else // for debug
        {
            foreach(i; 0 .. count)
                task(taskContext, 1, 0, i);
        }
    };
    if (auto err = optimizeImpl!(f, null, tm, T)(lm))
        throw new LeastSquaresLMException!T(err, lm);
}

/// With Jacobian
unittest
{
    import mir.ndslice.allocation: slice;
    import mir.ndslice.slice: sliced;
    import mir.blas: nrm2;

    auto lm = LeastSquaresLM!double(2, 2);
    lm.x[] = [100, 100];
    lm.optimize!(
        (x, y)
        {
            y[0] = x[0];
            y[1] = 2 - x[1];
        },
        (x, J)
        {
            J[0, 0] = 1;
            J[0, 1] = 0;
            J[1, 0] = 0;
            J[1, 1] = -1;
        },
    );

    assert(nrm2((lm.x - [0, 2].sliced).slice) < 1e-8);
}

/// Using Jacobian finite difference approximation computed using in multiple threads.
unittest
{
    import mir.ndslice.allocation: slice;
    import mir.ndslice.slice: sliced;
    import mir.blas: nrm2;
    import std.parallelism: taskPool;

    auto lm = LeastSquaresLM!double(2, 2);
    lm.x[] = [-1.2, 1];
    lm.optimize!(
        (x, y) // Rosenbrock function
        {
            y[0] = 10 * (x[1] - x[0]^^2);
            y[1] = 1 - x[0];
        },
    )(taskPool);

    assert(nrm2((lm.x - [1, 1].sliced).slice) < 1e-8);
}

/// Rosenbrock
unittest
{
    import mir.ndslice.algorithm: all;
    import mir.ndslice.allocation: slice;
    import mir.ndslice.slice: sliced;
    import mir.blas: nrm2;

    auto lm = LeastSquaresLM!double(2, 2, Yes.lowerBounds, Yes.upperBounds);
    lm.x[] = [-1.2, 1];

    alias rosenbrockRes = (x, y)
    {
        y[0] = 10 * (x[1] - x[0]^^2);
        y[1] = 1 - x[0];
    };

    alias rosenbrockJac = (x, J)
    {
        J[0, 0] = -20 * x[0];
        J[0, 1] = 10;
        J[1, 0] = -1;
        J[1, 1] = 0;
    };

    static class FFF
    {
        static auto opCall(ContiguousVector!(const double) x, ContiguousMatrix!double J)
        {
            rosenbrockJac(x, J);
        }
    }

    lm.optimize!(rosenbrockRes, FFF);

    // writeln(lm.iterCt, " ", lm.fCalls, " ", lm.gCalls);

    assert(nrm2((lm.x - [1, 1].sliced).slice) < 1e-8);

    /////

    lm.reset;
    lm.lower[] = [10.0, 10.0];
    lm.upper[] = [200.0, 200.0];
    lm.x[] = [150.0, 150.0];

    lm.optimize!(rosenbrockRes, rosenbrockJac);

    // writeln(lm.iterCt, " ", lm.fCalls, " ", lm.gCalls);
    assert(nrm2((lm.x - [10, 10].sliced).slice) < 1e-8);
    assert(lm.x.all!"a >= 10");
}

///
unittest
{
    import mir.blas: nrm2;
    import mir.math.common;
    import mir.ndslice.allocation: slice;
    import mir.ndslice.topology: linspace, map;
    import mir.ndslice.slice: sliced;
    import mir.random;
    import mir.random.algorithm;
    import mir.random.variable;
    import std.parallelism: taskPool;

    alias model = (x, p) => p[0] * map!exp(-x * p[1]);

    auto p = [1.0, 2.0];

    auto xdata = [20].linspace([0.0, 10.0]);
    auto rng = Random(12345);
    auto ydata = slice(model(xdata, p) + 0.01 * rng.randomSlice(normalVar, xdata.shape));

    auto lm = LeastSquaresLM!double(xdata.length, 2);
    lm.x[] = [0.5, 0.5];

    lm.optimize!((p, y) => y[] = model(xdata, p) - ydata)();

    assert((lm.x - [1.0, 2.0].sliced).slice.nrm2 < 0.05);
}

///
@safe pure unittest
{
    import mir.ndslice.algorithm: all;
    import mir.ndslice.allocation: slice;
    import mir.ndslice.topology: map, repeat, iota;
    import mir.ndslice.slice: sliced;
    import mir.random;
    import mir.random.variable;
    import mir.random.algorithm;
    import mir.math.common;

    alias model = (x, p) => p[0] * map!exp(-x / p[1]) + p[2];

    auto xdata = iota([100], 1);
    auto rng = Random(12345);
    auto ydata = slice(model(xdata, [10.0, 10.0, 10.0]) + 0.1 * rng.randomSlice(normalVar, xdata.shape));

    auto lm = LeastSquaresLM!double(xdata.length, 3, [5.0, 11.0, 5.0], double.infinity.repeat(3).slice.field);

    lm.x[] = [15.0, 15.0, 15.0];
    lm.optimize!((p, y) => 
        y[] = model(xdata, p) - ydata);

    assert(all!"a >= b"(lm.x, lm.lower));

    // writeln(lm.x);
    // writeln(lm);

    lm.reset;
    lm.x[] = [5.0, 5.0, 5.0];
    lm.upper[] = [15.0, 9.0, 15.0];
    lm.optimize!((p, y) => y[] = model(xdata, p) - ydata);

    assert(all!"a <= b"(lm.x, lm.upper));

    // writeln(lm.x);
    // writeln(lm);
}


/++
High level nothtow D API for Levenberg-Marquardt Algorithm.

Computes the argmin over x of `sum_i(f(x_i)^2)` using the Modified Levenberg-Marquardt
algorithm, and an estimate of the Jacobian of `f` at x.

The function `f` should take an input vector of length `n`, and fill an output
vector of length `m`.

The function `g` is the Jacobian of `f`, and should fill a row-major `m x n` matrix. 

Returns: optimization status.
Params:
    f = `n -> m` function
    g = `m × n` Jacobian (optional)
    tm = thread manager for finite difference jacobian approximation in case of g is null (optional)
    lm = Levenberg-Marquardt data structure
See_also: $(LREF optimize)
+/
LMStatus optimizeImpl(alias f, alias g = null, alias tm = null, T)(scope ref LeastSquaresLM!T lm)
{
    auto fInst = delegate(ContiguousVector!(const T) x, ContiguousVector!T y)
    {
        f(x, y);
    };
    if (false) with(lm)
        fInst(x, y);
    static if (is(typeof(g) == typeof(null)))
        enum LeastSquaresLM!T.JacobianDelegate gInst = null;
    else
    {
        auto gInst = delegate(ContiguousVector!(const T) x, ContiguousMatrix!T J)
        {
            g(x, J);
        };
        static if (isNullableFunction!(g))
            if (!g)
                gInst = null;
        if (false) with(lm)
            gInst(x, J);
    }

    static if (is(typeof(tm) == typeof(null)))
        enum LeastSquaresThreadManagerDelegate tmInst = null;
    else
    {
        auto tmInst = delegate(
            size_t count,
            void* taskContext,
            scope LeastSquaresTask task)
        {
            tm(count, taskContext, task);
        };
        // auto tmInst = &tmInstDec;
        static if (isNullableFunction!(tm))
            if (!tm)
                tmInst = null;
        if (false) with(lm)
            tmInst(0, null, null);
    }
    alias TM = typeof(tmInst);
    return optimizeLeastSquaresLM!T(lm, fInst.trustedAllAttr, gInst.trustedAllAttr,  tmInst.trustedAllAttr);
}

// extern (C) void delegate(ulong count, void* taskContext, extern (C) void function(void*, ulong, ulong, ulong) pure nothrow @nogc @safe task) @system 
// extern (C) void delegate(ulong count, void* taskContext, extern (C) void function(void*, ulong, ulong, ulong) pure nothrow @nogc @safe task) pure nothrow @nogc @safe

// extern (C) void delegate(ulong count, void* taskContext, scope extern (C) void function(void*, ulong, ulong, ulong) pure nothrow @nogc @safe task) @system 
// extern (C) void delegate(ulong count, void* taskContext, extern (C) void function(void* context, ulong totalThreads, ulong treadId, ulong i) pure nothrow @nogc @safe task) pure nothrow @nogc @safe
// void delegate(ulong, void*, scope extern (C) void function(void*, ulong, ulong, ulong) pure nothrow @nogc @safe) pure nothrow @nogc @safe to parameter scope extern (C) void delegate(ulong count, void* taskContext, extern (C) void function(void* context, ulong totalThreads, ulong treadId, ulong i) pure nothrow @nogc @safe task) pure nothrow @nogc @safe tm= cast(extern (C) void delegate(ulong count, void* taskContext, extern (C) void function(void* context, ulong totalThreads, ulong treadId, ulong i) pure nothrow @nogc @safe task) pure nothrow @nogc @safe)null

// optimizeLeastSquaresLMD(scope extern (C) void delegate(ulong count, void* taskContext, scope extern (C) void function(void* context, ulong totalThreads, ulong treadId, ulong i) pure nothrow @nogc @safe task) pure nothrow @nogc @safe tm = cast(extern (C) void delegate(ulong count, void* taskContext, scope extern (C) void function(void* context, ulong totalThreads,ulong treadId, ulong i) pure nothrow @nogc @safe task) pure nothrow @nogc @safe)null) is not callable using argument types (LeastSquaresLM!double, void delegate(Slice!(cast(SliceKind)2, [1LU], const(double)*), Slice!(cast(SliceKind)2, [1LU], double*)) pure nothrow @nogc @safe, void delegate(Slice!(cast(SliceKind)2, [1LU], const(double)*), Slice!(cast(SliceKind)2, [2LU], double*)) pure nothrow @nogc @safe, void delegate(ulong, void*, scope void function(void*, ulong, ulong, ulong) pure nothrow @nogc @safe) pure nothrow @nogc @safe)
// source/mir/least_squares.d(613,36):        cannot pass argument trustedAllAttr(tmInst) of type void delegate(ulong, void*, scope void function(void*, ulong, ulong, ulong) pure nothrow @nogc @safe) pure nothrow @nogc @safe to parameter scope extern (C) void delegate(ulong count, void* taskContext, scope extern (C) void function(void* context, ulong totalThreads, ulong treadId, ulong i) pure nothrow @nogc @safe task) pure nothrow @nogc @safe tm = cas

// void delegate(
//     Slice!(cast(SliceKind)2, [1LU], const(double)*),
//     Slice!(cast(SliceKind)2, [1LU], double*)) pure nothrow @nogc @safe,
//     void delegate (
//         Slice!(cast(SliceKind)2, [1LU], const(double)*), Slice!(cast(SliceKind)2, [2LU], double*)) pure nothrow @nogc @safe, void delegate(ulong, void*, scope void function(void*, ulong, ulong, ulong) pure nothrow @nogc @safe) pure nothrow @nogc @safe)

///
enum LMStatus : byte
{
    ///
    success = 0,
    ///
    initialized,
    ///
    badBounds = -32,
    ///
    badGuess,
    ///
    badMinStepQuality,
    ///
    badGoodStepQuality,
    ///
    badStepQuality,
    ///
    badLambdaParams,
}

/++
Status string for low (extern) and middle (nothrow) levels D API.
Params:
    st = optimization status
Returns: description for $(LMStatus)
+/
pragma(inline, false)
string lmStatusString(LMStatus st) @safe pure nothrow @nogc
{
    final switch(st) with(LMStatus)
    {
        case success:
            return "success";
        case initialized:
            return "data structure was initialized";
        case badBounds:
            return "bounds must either be empty or of the same as argument number of parameters.";
        case badGuess:
            return "Initial guess must be within bounds.";
        case badMinStepQuality:
            return "0 <= minStepQuality < 1 must hold.";
        case badGoodStepQuality:
            return "0 < goodStepQuality <= 1 must hold.";
        case badStepQuality:
            return "minStepQuality < goodStepQuality must hold.";
        case badLambdaParams:
            return "1 <= lambdaIncrease && lambdaIncrease <= T.max.sqrt and T.min_normal.sqrt <= lambdaDecrease && lambdaDecrease <= 1 must hold.";
    }
}

///
alias LeastSquaresTask = extern(C) void function(
                void* context,
                size_t totalThreads,
                size_t treadId,
                size_t i)
            @safe nothrow @nogc pure;

/// Thread manager delegate type for low level `extern(D)` API.
alias LeastSquaresThreadManagerDelegate = void delegate(
        size_t count,
        void* taskContext,
        scope LeastSquaresTask task,
        )@safe nothrow @nogc pure;

/++
Low level `extern(D)` instatiation.
Params:
    lm = Levenberg-Marquardt data structure
    f = `n -> m` function
    g = `m × n` Jacobian (optional)
    tm = thread manager for finite difference jacobian approximation in case of g is null (optional)
+/
pragma(inline, false)
LMStatus optimizeLeastSquaresLMD
    (
        scope ref LeastSquaresLM!double lm,
        scope LeastSquaresLM!double.FunctionDelegate f,
        scope LeastSquaresLM!double.JacobianDelegate g = null,
        scope LeastSquaresThreadManagerDelegate tm = null,
    ) @trusted nothrow @nogc pure
{
    return optimizeLMImplGeneric!double(lm, f, g, tm);
}


/// ditto
pragma(inline, false)
LMStatus optimizeLeastSquaresLMS
    (
        scope ref LeastSquaresLM!float lm,
        scope LeastSquaresLM!float.FunctionDelegate f,
        scope LeastSquaresLM!float.JacobianDelegate g = null,
        scope LeastSquaresThreadManagerDelegate tm = null,
    ) @trusted nothrow @nogc pure
{
    return optimizeLMImplGeneric!float(lm, f, g, tm);
}

/// ditto
alias optimizeLeastSquaresLM(T : double) = optimizeLeastSquaresLMD;
/// ditto
alias optimizeLeastSquaresLM(T : float) = optimizeLeastSquaresLMS;


extern(C) @safe nothrow @nogc
{
    /++
    Status string for extern(C) API.
    Params:
        st = optimization status
    Returns: description for $(LMStatus)
    +/
    extern(C)
    pragma(inline, false)
    immutable(char)* mir_least_squares_lm_status_string(LMStatus st) @trusted pure nothrow @nogc
    {
        return st.lmStatusString.ptr;
    }

    /// Thread manager function type for low level `extern(C)` API.
    alias LeastSquaresThreadManagerFunction =
        extern(C) void function(
            void* context,
            size_t count,
            void* taskContext,
            scope LeastSquaresTask task)
            @system nothrow @nogc pure;

    /++
    Low level `extern(C)` wrapper instatiation.
    Params:
        lm = Levenberg-Marquardt data structure
        fContext = context for the function
        f = `n -> m` function
        gContext = context for the Jacobian (optional)
        g = `m × n` Jacobian (optional)
        tm = thread manager for finite difference jacobian approximation in case of g is null (optional)
    +/
    extern(C)
    pragma(inline, false)
    LMStatus mir_least_squares_lm_optimize_d
        (
            scope ref LeastSquaresLM!double lm,
            scope void* fContext,
            scope LeastSquaresLM!double.FunctionFunction f,
            scope void* gContext = null,
            scope LeastSquaresLM!double.JacobianFunction g = null,
            scope void* tmContext = null,
            scope LeastSquaresThreadManagerFunction tm = null,
        ) @system nothrow @nogc pure
    {
        return optimizeLMImplGenericBetterC!double(lm, fContext, f, gContext, g, tmContext, tm);
    }

    /// ditto
    extern(C)
    pragma(inline, false)
    LMStatus mir_least_squares_lm_optimize_s
        (
            scope ref LeastSquaresLM!float lm,
            scope void* fContext,
            scope LeastSquaresLM!float.FunctionFunction f,
            scope void* gContext = null,
            scope LeastSquaresLM!float.JacobianFunction g = null,
            scope void* tmContext = null,
            scope LeastSquaresThreadManagerFunction tm = null,
        ) @system nothrow @nogc pure
    {
        return optimizeLMImplGenericBetterC!float(lm, fContext, f, gContext, g, tmContext, tm);
    }

    /// ditto
    alias mir_least_squares_lm_optimize(T : double) = mir_least_squares_lm_optimize_d;

    /// ditto
    alias mir_least_squares_lm_optimize(T : float) = mir_least_squares_lm_optimize_s;

    /++
    Initialize LM data structure with default params for iteration.
    Params:
        lm = Levenberg-Marquart data structure
    +/
    void mir_least_squares_lm_init_params_d(ref LeastSquaresLM!double lm) pure
    {
        lm.initParams;
    }

    /// ditto
    void mir_least_squares_lm_init_params_s(ref LeastSquaresLM!float lm) pure
    {
        lm.initParams;
    }

    /// ditto
    alias mir_least_squares_lm_init_params(T : double) = mir_least_squares_lm_init_params_d;

    /// ditto
    alias mir_least_squares_lm_init_params(T : float) = mir_least_squares_lm_init_params_s;

    /++
    Resets all counters and flags, fills `x`, `y`, `upper`, `lower`, vecors with default values.
    Params:
        lm = Levenberg-Marquart data structure
    +/
    void mir_least_squares_lm_reset_d(ref LeastSquaresLM!double lm) pure
    {
        lm.reset;
    }

    /// ditto
    void mir_least_squares_lm_reset_s(ref LeastSquaresLM!float lm) pure
    {
        lm.reset;
    }

    /// ditto
    alias mir_least_squares_lm_reset(T : double) = mir_least_squares_lm_reset_d;

    /// ditto
    alias mir_least_squares_lm_reset(T : float) = mir_least_squares_lm_reset_s;

    /++
    Allocates data.
    Params:
        lm = Levenberg-Marquart data structure
        m = Y = f(X) dimension
        n = X dimension
        lowerBounds = flag to allocate lower bounds
        lowerBounds = flag to allocate upper bounds
    +/
    void mir_least_squares_lm_stdc_alloc_d(ref LeastSquaresLM!double lm, size_t m, size_t n, bool lowerBounds, bool upperBounds)
    {
        lm.stdcAlloc(m, n, lowerBounds, upperBounds);
    }

    /// ditto
    void mir_least_squares_lm_stdc_alloc_s(ref LeastSquaresLM!float lm, size_t m, size_t n, bool lowerBounds, bool upperBounds)
    {
        lm.stdcAlloc(m, n, lowerBounds, upperBounds);
    }

    /// ditto
    alias mir_least_squares_lm_stdc_alloc(T : double) = mir_least_squares_lm_stdc_alloc_d;

    /// ditto
    alias mir_least_squares_lm_stdc_alloc(T : float) = mir_least_squares_lm_stdc_alloc_s;

    /++
    Frees vectors including `x`, `y`, `upper`, `lower`.
    Params:
        lm = Levenberg-Marquart data structure
    +/
    void mir_least_squares_lm_stdc_free_d(ref LeastSquaresLM!double lm)
    {
        lm.stdcFree;
    }

    /// ditto
    void mir_least_squares_lm_stdc_free_s(ref LeastSquaresLM!float lm)
    {
        lm.stdcFree;
    }

    /// ditto
    alias mir_least_squares_lm_stdc_free(T : double) = mir_least_squares_lm_stdc_free_d;

    /// ditto
    alias mir_least_squares_lm_stdc_free(T : float) = mir_least_squares_lm_stdc_free_s;
}

private:

LMStatus optimizeLMImplGenericBetterC(T)
    (
        scope ref LeastSquaresLM!T lm,
        scope void* fContext,
        scope LeastSquaresLM!T.FunctionFunction f,
        scope void* gContext,
        scope LeastSquaresLM!T.JacobianFunction g,
        scope void* tmContext,
        scope LeastSquaresThreadManagerFunction tm,
    ) @system nothrow @nogc pure
{
    version(LDC) pragma(inline, true);
    if (g)
        return optimizeLeastSquaresLM!T(
            lm,
            (x, y) @trusted => f(fContext, y.length, x.length, x.iterator, y.iterator),
            (x, J) @trusted => g(gContext, J.length, x.length, x.iterator, J.iterator),
            null
        );
    if (tm)
        return optimizeLeastSquaresLM!T(
            lm,
            (x, y) @trusted => f(fContext, y.length, x.length, x.iterator, y.iterator),
            null, (count, taskContext, scope task)  @trusted => tm(tmContext, count, taskContext, task)
        );
    return optimizeLeastSquaresLM!T(
        lm,
        (x, y) @trusted => f(fContext, y.length, x.length, x.iterator, y.iterator),
        null,
        null
    );
}

extern(C) void defaultLMThreadManagerDelegate(T)(void* context, size_t totalThreads, size_t treadId, size_t j) @trusted pure nothrow @nogc
{with(*cast(LeastSquaresLM!T*)((cast(void**)context)[0])){
    import mir.blas;
    import mir.math.common;
    auto f = *cast(LeastSquaresLM!T.FunctionDelegate*)((cast(void**)context)[1]);
    auto idx = totalThreads >= n ? j : treadId;
    auto p = JJ[idx];
    if(ipiv[idx]++ == 0)
    {
        copy(x, p);
    }

    auto save = p[j];
    auto xmh = save - jacobianEpsilon;
    auto xph = save + jacobianEpsilon;
    if (_lower_ptr)
        xmh = fmax(xmh, lower[j]);
    if (_upper_ptr)
        xph = fmin(xph, upper[j]);
    auto Jj = J[0 .. $, j];
    if (auto twh = xph - xmh)
    {
        p[j] = xph;
        f(p, mBuffer);
        copy(mBuffer, Jj);

        p[j] = xmh;
        f(p, mBuffer);

        p[j] = save;

        axpy(-1, mBuffer, Jj);
        scal(1 / twh, Jj);
    }
    else
    {
        fill(0, Jj);
    }
}}

// LM algorithm
LMStatus optimizeLMImplGeneric(T)
    (
        scope ref LeastSquaresLM!T lm,
        scope LeastSquaresLM!T.FunctionDelegate f,
        scope LeastSquaresLM!T.JacobianDelegate g = null,
        scope LeastSquaresThreadManagerDelegate tm = null,
    ) @trusted nothrow @nogc
{with(lm){
    import mir.blas;
    import mir.lapack;
    import mir.math.common;
    import mir.math.sum: sum;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.topology: canonical, diagonal;
    import mir.utility: max;
    import mir.ndslice.slice: sliced;

    version(LDC) pragma(inline, true);

    if (_lower_ptr && lower.length != x.length || _upper_ptr && upper.length != x.length)
        return LMStatus.badBounds;
    if (!(!_lower_ptr || allLessOrEqual(lower, x)) || !(!_upper_ptr || allLessOrEqual(x, upper)))
        return LMStatus.badGuess;
    if (!(0 <= minStepQuality && minStepQuality < 1))
        return LMStatus.badMinStepQuality;
    if (!(0 <= goodStepQuality && goodStepQuality <= 1))
        return LMStatus.badGoodStepQuality;
    if (!(minStepQuality < goodStepQuality))
        return LMStatus.badStepQuality;
    if (!(1 <= lambdaIncrease && lambdaIncrease <= T.max.sqrt))
        return LMStatus.badLambdaParams;
    if (!(T.min_normal.sqrt <= lambdaDecrease && lambdaDecrease <= 1))
        return LMStatus.badLambdaParams;

    maxAge = maxAge ? maxAge : g ? 3 : cast(uint)(2 * n);
    uint age = maxAge;
    
    tm = tm ? tm : delegate(size_t count, void* taskContext, scope LeastSquaresTask task) pure @nogc nothrow @trusted
    {
        foreach(i; 0 .. count)
            task(taskContext, 1, 0, i);
    };

    bool needJacobian = true;
    f(x, y);
    fCalls += 1;
    residual = dot(y, y);

    do
    {
        if (needJacobian)
        {
            needJacobian = false;
            if (age < maxAge)
            {
                age++;
                auto d = 1 / dot(deltaXBase, deltaXBase);
                axpy(-1, y, mBuffer); // -deltaY
                gemv(1, J, deltaXBase, 1, mBuffer); //-(f_new - f_old - J_old*h)
                scal(-d, mBuffer);
                ger(1, mBuffer, deltaXBase, J); //J_new = J_old + u*h'
            }
            else
            if (g)
            {
                age = 0;
                g(x, J);
                gCalls += 1;
            }
            else
            {
                age = 0;
                fill(0, ipiv);
                void*[2] context;
                context[0] = &lm;
                context[1] = &f;
                tm(n, context.ptr, &defaultLMThreadManagerDelegate!T);
                fCalls += ipiv.sum;
            }
            gemv(-1, J.transposed, y, 0, mJy);
        }

        syrk(Uplo.Upper, 1, J.transposed, 0, JJ);
        foreach(ref e; JJ.diagonal)
        {
            e += lambda * fmax(e, minDiagonal);
        }
        copy(mJy, deltaX);
        sysv_rook('L', JJ.canonical, ipiv, deltaX.sliced([1, n]).canonical, _work);
        axpy(1, x, deltaX);

        if (_lower_ptr)
        {
            foreach(i; 0 .. n)
            {
                deltaX[i] = deltaX[i].fmax(lower[i]);
            }
        }
        if (_upper_ptr)
        {
            foreach(i; 0 .. n)
            {
                deltaX[i] = deltaX[i].fmin(upper[i]);
            }
        }

        axpy(-1, x, deltaX);
        copy(y, mBuffer);
        gemv(1, J, deltaX, 1, mBuffer);
        auto predictedResidual = dot(mBuffer, mBuffer);
        auto nBuffer = JJ[0];
        copy(x, nBuffer);
        axpy(1, deltaX, nBuffer);
        f(nBuffer, mBuffer);
        fCalls += 1;
        auto trialResidual = dot(mBuffer, mBuffer);
        auto rho = (trialResidual - residual) / (predictedResidual - residual);

        if (rho > minStepQuality)
        {
            copy(deltaX, deltaXBase);
            copy(nBuffer, x);
            swap(y, mBuffer);
            residual = trialResidual;
            if (rho > goodStepQuality)
            {
                lambda = fmax(lambdaDecrease * lambda, minLambda);
            }
            needJacobian = true;
        }
        else
        {
            lambda = fmin(lambdaIncrease * lambda, maxLambda);
        }

        gemv(1, J.transposed, y, 0, nBuffer);
        gConverged = nBuffer.amax < tolG;
        xConverged = deltaX.nrm2 < tolX * (tolX + x.nrm2);

        if (age && (gConverged || xConverged))
        {
            gConverged = false;
            xConverged = false;
            age = maxAge;
            needJacobian = true;
        }

        ++iterCt;
    }
    while (iterCt < maxIter && !(gConverged || xConverged));

    return LMStatus.success;
}}

pragma(inline, false)
T amax(T)(Slice!(Contiguous, [1], const(T)*) x)
{
    import mir.math.common: fmax, fabs;
    T ret = 0;
    foreach(ref e; x.field)
        ret = fmax(fabs(e), ret);
    return ret;
}

pragma(inline, false)
void fill(T, SliceKind kind)(T value, Slice!(kind, [1], T*) x)
{
    x[] = value;
}

pragma(inline, false)
bool allLessOrEqual(T, SliceKind kind)(
    Slice!(kind, [1], const(T)*) a,
    Slice!(kind, [1], const(T)*) b,
    )
{
    import mir.ndslice.algorithm: all;
    return all!"a <= b"(a, b);
}

uint normalizeSafety()(uint attrs)
{
    if (attrs & FunctionAttribute.system)
        attrs &= ~FunctionAttribute.safe;
    return attrs;
}

auto trustedAllAttr(T)(T t) @trusted
    if (isFunctionPointer!T || isDelegate!T)
{
    enum attrs = (functionAttributes!T & ~FunctionAttribute.system) 
        | FunctionAttribute.pure_
        | FunctionAttribute.safe
        | FunctionAttribute.nogc
        | FunctionAttribute.nothrow_;
    return cast(SetFunctionAttributes!(T, functionLinkage!T, attrs)) t;
}

template isNullableFunction(alias f)
{
    enum isNullableFunction = __traits(compiles, { alias F = Unqual!(typeof(f)); auto r = function(ref F e) {e = null;};} );
}
