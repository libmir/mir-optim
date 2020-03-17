/++
$(H1 Nonlinear Least Squares Solver)

Copyright: Copyright © 2018, Symmetry Investments & Kaleidic Associates Advisory Limited
Authors:   Ilya Yaroshenko

Macros:
NDSLICE = $(REF_ALTTEXT $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
+/
module mir.optim.least_squares;

import mir.ndslice.slice: Slice, SliceKind, Contiguous, sliced;
import std.meta;
import std.traits;
  
public import std.typecons: Flag, Yes, No;

version = mir_optim_test;

///
enum LMStatus
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
    ///
    numericError,
}

version(D_Exceptions)
{
    /+
    Exception for $(LREF optimize).
    +/
    private static immutable leastSquaresLMException_initialized = new Exception("mir-optim LM-algorithm: status is 'initialized', zero iterations");
    private static immutable leastSquaresLMException_badBounds = new Exception("mir-optim LM-algorithm: " ~ LMStatus.badBounds.lmStatusString);
    private static immutable leastSquaresLMException_badGuess = new Exception("mir-optim LM-algorithm: " ~ LMStatus.badGuess.lmStatusString);
    private static immutable leastSquaresLMException_badMinStepQuality = new Exception("mir-optim LM-algorithm: " ~ LMStatus.badMinStepQuality.lmStatusString);
    private static immutable leastSquaresLMException_badGoodStepQuality = new Exception("mir-optim LM-algorithm: " ~ LMStatus.badGoodStepQuality.lmStatusString);
    private static immutable leastSquaresLMException_badStepQuality = new Exception("mir-optim LM-algorithm: " ~ LMStatus.badStepQuality.lmStatusString);
    private static immutable leastSquaresLMException_badLambdaParams = new Exception("mir-optim LM-algorithm: " ~ LMStatus.badLambdaParams.lmStatusString);
    private static immutable leastSquaresLMException_numericError = new Exception("mir-optim LM-algorithm: " ~ LMStatus.numericError.lmStatusString);
    private static immutable leastSquaresLMExceptions = [
        leastSquaresLMException_initialized,
        leastSquaresLMException_badBounds,
        leastSquaresLMException_badGuess,
        leastSquaresLMException_badMinStepQuality,
        leastSquaresLMException_badGoodStepQuality,
        leastSquaresLMException_badStepQuality,
        leastSquaresLMException_badLambdaParams,
        leastSquaresLMException_numericError,
    ];
}

/++
Modified Levenberg-Marquardt parameters, data, and state.
+/
struct LeastSquaresLM(T)
    if (is(T == double) || is(T == float))
{

    import mir.optim.boxcqp;
    import mir.math.common: sqrt;
    import mir.math.constant: GoldenRatio;
    import lapack: lapackint;

    /// Default tolerance in x
    enum T tolXDefault = T.epsilon;;
    /// Default tolerance in gradient
    enum T tolGDefault = T.epsilon; //T(2) ^^ ((1 - T.mant_dig) * 3 / 4);
    /// Default value for `maxGoodResidual`.
    enum T maxGoodResidualDefault = T.epsilon;
    /// Default epsilon for finite difference Jacobian approximation
    enum T jacobianEpsilonDefault = T(2) ^^ ((1 - T.mant_dig) / 2);
    /// Default `lambda` is multiplied by this factor after step below min quality
    enum T lambdaIncreaseDefault = 4;
    /// Default `lambda` is multiplied by this factor after good quality steps
    enum T lambdaDecreaseDefault = 1 / (GoldenRatio * lambdaIncreaseDefault);
    /// Default scale such as for steps below this quality, the trust region is shrinked
    enum T minStepQualityDefault = 0.1;
    /// Default scale such as for steps above thsis quality, the trust region is expanded
    enum T goodStepQualityDefault = 0.68;
    /// Default maximum trust region radius
    enum T maxLambdaDefault = T.max / 16;
    /// Default maximum trust region radius
    enum T minLambdaDefault = T.min_normal * 16;

    /// Delegates for low level D API.
    alias Function = void delegate(Slice!(const(T)*) x, Slice!(T*) y) @safe nothrow @nogc pure;
    /// ditto
    alias Jacobian = void delegate(Slice!(const(T)*) x, Slice!(T*, 2) J) @safe nothrow @nogc pure;

    /// Delegates for low level C API.
    alias FunctionBetterC = extern(C) void function(scope void* context, size_t m, size_t n, const(T)* x, T* y) @system nothrow @nogc pure;
    ///
    alias JacobianBetterC = extern(C) void function(scope void* context, size_t m, size_t n, const(T)* x, T* J) @system nothrow @nogc pure;

    /// maximum number of iterations
    uint maxIter = 1000;
    /// tolerance in x
    T tolX = tolXDefault;
    /// tolerance in gradient
    T tolG = tolGDefault;
    /// the algorithm stops iteration when the residual value is less or equal to `maxGoodResidual`.
    T maxGoodResidual = maxGoodResidualDefault;
    /// `lambda` is multiplied by this factor after step below min quality
    T lambdaIncrease = lambdaIncreaseDefault;
    /// `lambda` is multiplied by this factor after good quality steps
    T lambdaDecrease = lambdaDecreaseDefault;
    /// for steps below this quality, the trust region is shrinked
    T minStepQuality = minStepQualityDefault;
    /// for steps above this quality, the trust region is expanded
    T goodStepQuality = goodStepQualityDefault;
    /// minimum trust region radius
    T maxLambda = maxLambdaDefault;
    /// maximum trust region radius
    T minLambda = minLambdaDefault;
    /// epsilon for finite difference Jacobian approximation
    T jacobianEpsilon = jacobianEpsilonDefault;

    /++
    Counters and state values.
    +/
    size_t iterCt;
    /// ditto
    size_t fCalls;
    /// ditto
    size_t gCalls;
    /// ditto
    T residual = T.infinity;
    /// (inverse of) initial trust region radius
    T lambda = 0;
    /// ditto
    uint maxAge;
    /// ditto
    LMStatus status = LMStatus.initialized;
    /// ditto
    bool xConverged;
    /// ditto
    bool gConverged;
    /// ditto
    /// `residual <= maxGoodResidual`
    bool fConverged()() const @property
    {
        return residual <= maxGoodResidual;
    }

    /++
    Resets all counters and flags, fills `x`, `y`, `upper`, `lower`, vecors with default values.
    +/
    pragma(inline, false)
    void reset()() @safe pure nothrow @nogc
    {
        lambda = 0;
        iterCt = 0;
        fCalls = 0;
        gCalls = 0;     
        residual = T.infinity;
        maxAge = 0;
        status = LMStatus.initialized;
        xConverged = false;
        gConverged = false;    
    }
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
void optimize(alias f, alias g = null, alias tm = null, T)(
    scope ref LeastSquaresLM!T lm,
    size_t m,
    Slice!(T*) x,
    Slice!(const(T)*) l,
    Slice!(const(T)*) u,
)
    if ((is(T == float) || is(T == double)))
{
    if (auto err = optimizeImpl!(f, g, tm, T)(lm, m, x, l, u))
        throw leastSquaresLMExceptions[err == 1 ? 0 : err + 33];
}

/// ditto
void optimize(alias f, TaskPool, T)(
    scope ref LeastSquaresLM!T lm,
    size_t m,
    Slice!(T*) x,
    Slice!(const(T)*) l,
    Slice!(const(T)*) u,
    TaskPool taskPool)
    if (is(T == float) || is(T == double))
{
    auto tm = delegate(uint count, scope LeastSquaresTask task)
    {
        version(all)
        {
            import mir.ndslice.topology: iota;
            foreach(i; taskPool.parallel(count.iota!uint))
                task(cast(uint)taskPool.size, cast(uint)(taskPool.size <= 1 ? 0 : taskPool.workerIndex - 1), i);
        }
        else // for debug
        {
            foreach(i; 0 .. count)
                task(1, 0, i);
        }
    };
    if (auto err = optimizeImpl!(f, null, tm, T)(lm, m, x, l, u))
        throw leastSquaresLMExceptions[err == 1 ? 0 : err + 33];
}

/// With Jacobian
version(mir_optim_test)
@safe unittest
{
    import mir.ndslice.allocation: slice;
    import mir.ndslice.slice: sliced;
    import mir.blas: nrm2;

    LeastSquaresLM!double lm;
    auto x = [100.0, 100].sliced;
    auto l = x.shape.slice(-double.infinity);
    auto u = x.shape.slice(+double.infinity);
    optimize!(
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
    )(lm, 2, x, l, u);

    assert(nrm2((x - [0, 2].sliced).slice) < 1e-8);
}

/// Using Jacobian finite difference approximation computed using in multiple threads.
version(mir_optim_test)
unittest
{
    import mir.ndslice.allocation: slice;
    import mir.ndslice.slice: sliced;
    import mir.blas: nrm2;
    import std.parallelism: taskPool;

    LeastSquaresLM!double lm;
    auto x = [-1.2, 1].sliced;
    auto l = x.shape.slice(-double.infinity);
    auto u = x.shape.slice(+double.infinity);
    lm.optimize!(
        (x, y) // Rosenbrock function
        {
            y[0] = 10 * (x[1] - x[0]^^2);
            y[1] = 1 - x[0];
        },
    )(2, x, l, u, taskPool);

    assert(nrm2((x - [1, 1].sliced).slice) < 1e-6);
}

/// Rosenbrock
version(mir_optim_test)
@safe unittest
{
    import mir.algorithm.iteration: all;
    import mir.ndslice.allocation: slice;
    import mir.ndslice.slice: sliced;
    import mir.blas: nrm2;

    LeastSquaresLM!double lm;
    auto x = [-1.2, 1].sliced;
    auto l = x.shape.slice(-double.infinity);
    auto u = x.shape.slice(+double.infinity);

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
        static auto opCall(Slice!(const(double)*) x, Slice!(double*, 2) J)
        {
            rosenbrockJac(x, J);
        }
    }

    lm.optimize!(rosenbrockRes, FFF)(2, x, l, u);

    // import std.stdio;

    // writeln(lm.iterCt, " ", lm.fCalls, " ", lm.gCalls, " x = ", x);

    assert(nrm2((x - [1, 1].sliced).slice) < 1e-8);

    /////

    lm.reset;
    x[] = [150.0, 150.0];
    l[] = [10.0, 10.0];
    u[] = [200.0, 200.0];

    lm.optimize!(rosenbrockRes, rosenbrockJac)(2, x, l, u);

    // writeln(lm.iterCt, " ", lm.fCalls, " ", lm.gCalls, " ", x);
    assert(nrm2((x - [10, 100].sliced).slice) < 1e-5);
    assert(x.all!"a >= 10");
}

///
version(mir_optim_test)
@safe unittest
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

    auto x = [0.5, 0.5].sliced;
    auto l = x.shape.slice(-double.infinity);
    auto u = x.shape.slice(+double.infinity);

    LeastSquaresLM!double lm;
    lm.optimize!((p, y) => y[] = model(xdata, p) - ydata)(ydata.length, x, l, u);

    assert((x - [1.0, 2.0].sliced).slice.nrm2 < 0.05);
}

///
version(mir_optim_test)
@safe pure unittest
{
    import mir.algorithm.iteration: all;
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

    LeastSquaresLM!double lm;

    auto x = [15.0, 15.0, 15.0].sliced;
    auto l = [5.0, 11.0, 5.0].sliced;
    auto u = x.shape.slice(+double.infinity);

    lm.optimize!((p, y) => y[] = model(xdata, p) - ydata)
        (ydata.length, x, l, u);

    assert(all!"a >= b"(x, l));

    // import std.stdio;

    // writeln(x);
    // writeln(lm.iterCt, " ", lm.fCalls, " ", lm.gCalls);

    lm.reset;
    x[] = [5.0, 5.0, 5.0];
    l[] = -double.infinity;
    u[] = [15.0, 9.0, 15.0];
    lm.optimize!((p, y) => y[] = model(xdata, p) - ydata)
        (ydata.length, x, l , u);

    assert(x.all!"a <= b"(u));

    // writeln(x);
    // writeln(lm.iterCt, " ", lm.fCalls, " ", lm.gCalls);
}

///
version(mir_optim_test)
@safe pure unittest
{
    import mir.blas: nrm2;
    import mir.math.common: sqrt;
    import mir.ndslice.allocation: slice;
    import mir.ndslice.slice: sliced;

    LeastSquaresLM!double lm;
    auto x = [0.001, 0.0001].sliced;
    auto l = [-0.5, -0.5].sliced;
    auto u = [0.5, 0.5].sliced;
    lm.optimize!(
        (x, y)
        {
            y[0] = sqrt(1 - (x[0] ^^ 2 + x[1] ^^ 2));
        },
    )(1, x, l, u);

    assert(nrm2((x - u).slice) < 1e-8);
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
LMStatus optimizeImpl(alias f, alias g = null, alias tm = null, T)(
    scope ref LeastSquaresLM!T lm,
    size_t m,
    Slice!(T*) x,
    Slice!(const(T)*) l,
    Slice!(const(T)*) u,
)
{
    auto fInst = delegate(Slice!(const(T)*) x, Slice!(T*) y)
    {
        f(x, y);
    };
    if (false)
    {
        fInst(x, x);
    }
    static if (is(typeof(g) == typeof(null)))
        enum LeastSquaresLM!T.Jacobian gInst = null;
    else
    {
        auto gInst = delegate(Slice!(const(T)*) x, Slice!(T*, 2) J)
        {
            g(x, J);
        };
        static if (isNullableFunction!(g))
            if (!g)
                gInst = null;
        if (false)
        {
            Slice!(T*, 2) J;
            gInst(x, J);
        }
    }

    static if (is(typeof(tm) == typeof(null)))
        enum LeastSquaresThreadManager tmInst = null;
    else
    {
        auto tmInst = delegate(
            uint count,
            scope LeastSquaresTask task)
        {
            tm(count, task);
        };
        // auto tmInst = &tmInstDec;
        static if (isNullableFunction!(tm))
            if (!tm)
                tmInst = null;
        if (false) with(lm)
            tmInst(0, null);
    }
    alias TM = typeof(tmInst);
    return optimizeLeastSquaresLM!T(lm, m, x, l, u, fInst.trustedAllAttr, gInst.trustedAllAttr,  tmInst.trustedAllAttr);
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
            return "Initial guess must be within bounds.";
        case badGuess:
            return "Initial guess must be an array of finite numbers.";
        case badMinStepQuality:
            return "0 <= minStepQuality < 1 must hold.";
        case badGoodStepQuality:
            return "0 < goodStepQuality <= 1 must hold.";
        case badStepQuality:
            return "minStepQuality < goodStepQuality must hold.";
        case badLambdaParams:
            return "1 <= lambdaIncrease && lambdaIncrease <= T.max.sqrt and T.min_normal.sqrt <= lambdaDecrease && lambdaDecrease <= 1 must hold.";
        case numericError:
            return "numeric error";
    }
}

///
alias LeastSquaresTask = void delegate(
        uint totalThreads,
        uint threadId,
        uint i)
    @safe nothrow @nogc pure;

///
alias LeastSquaresTaskBetterC = extern(C) void function(
        scope ref const LeastSquaresTask,
        uint totalThreads,
        uint threadId,
        uint i)
    @safe nothrow @nogc pure;

/// Thread manager delegate type for low level `extern(D)` API.
alias LeastSquaresThreadManager = void delegate(
        uint count,
        scope LeastSquaresTask task)
    @safe nothrow @nogc pure;

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
        size_t m,
        Slice!(double*) x,
        Slice!(const(double)*) l,
        Slice!(const(double)*) u,
        scope LeastSquaresLM!double.Function f,
        scope LeastSquaresLM!double.Jacobian g = null,
        scope LeastSquaresThreadManager tm = null,
    ) @trusted nothrow @nogc pure
{
    return optimizeLMImplGeneric!double(lm, m, x, l, u, f, g, tm);
}


/// ditto
pragma(inline, false)
LMStatus optimizeLeastSquaresLMS
    (
        scope ref LeastSquaresLM!float lm,
        size_t m,
        Slice!(float*) x,
        Slice!(const(float)*) l,
        Slice!(const(float)*) u,
        scope LeastSquaresLM!float.Function f,
        scope LeastSquaresLM!float.Jacobian g = null,
        scope LeastSquaresThreadManager tm = null,
    ) @trusted nothrow @nogc pure
{
    return optimizeLMImplGeneric!float(lm, 2, x, l, u, f, g, tm);
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
    alias LeastSquaresThreadManagerBetterC =
        extern(C) void function(
            scope void* context,
            uint count,
            scope ref const LeastSquaresTask taskContext,
            scope LeastSquaresTaskBetterC task)
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
            uint m,
            uint n,
            double* x,
            const(double)* l,
            const(double)* u,
            scope void* fContext,
            scope LeastSquaresLM!double.FunctionBetterC f,
            scope void* gContext = null,
            scope LeastSquaresLM!double.JacobianBetterC g = null,
            scope void* tmContext = null,
            scope LeastSquaresThreadManagerBetterC tm = null,
        ) @system nothrow @nogc pure
    {
        return optimizeLMImplGenericBetterC!double(lm, m, n, x, l, u, fContext, f, gContext, g, tmContext, tm);
    }

    /// ditto
    extern(C)
    pragma(inline, false)
    LMStatus mir_least_squares_lm_optimize_s
        (
            scope ref LeastSquaresLM!float lm,
            uint m,
            uint n,
            float* x,
            const(float)* l,
            const(float)* u,
            scope void* fContext,
            scope LeastSquaresLM!float.FunctionBetterC f,
            scope void* gContext = null,
            scope LeastSquaresLM!float.JacobianBetterC g = null,
            scope void* tmContext = null,
            scope LeastSquaresThreadManagerBetterC tm = null,
        ) @system nothrow @nogc pure
    {
        return optimizeLMImplGenericBetterC!float(lm, m, n, x, l, u, fContext, f, gContext, g, tmContext, tm);
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
    void mir_least_squares_lm_init_d(ref LeastSquaresLM!double lm) pure
    {
        lm = lm.init;
    }

    /// ditto
    void mir_least_squares_lm_init_s(ref LeastSquaresLM!float lm) pure
    {
        lm = lm.init;
    }

    /// ditto
    alias mir_least_squares_lm_init(T : double) = mir_least_squares_lm_init_d;

    /// ditto
    alias mir_least_squares_lm_init(T : float) = mir_least_squares_lm_init_s;

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
}

private:

LMStatus optimizeLMImplGenericBetterC(T)
    (
        scope ref LeastSquaresLM!T lm,
        uint m,
        uint n,
        T* x,
        const(T)* l,
        const(T)* u,
        scope void* fContext,
        scope LeastSquaresLM!T.FunctionBetterC f,
        scope void* gContext,
        scope LeastSquaresLM!T.JacobianBetterC g,
        scope void* tmContext,
        scope LeastSquaresThreadManagerBetterC tm,
    ) @system nothrow @nogc pure
{
    version(LDC) pragma(inline, true);
    if (g)
        return optimizeLeastSquaresLM!T(
            lm,
            m,
            x[0 .. n].sliced,
            l[0 .. n].sliced,
            u[0 .. n].sliced,
            (x, y) @trusted => f(fContext, y.length, x.length, x.iterator, y.iterator),
            (x, J) @trusted => g(gContext, J.length, x.length, x.iterator, J.iterator),
            null
        );

    LeastSquaresTaskBetterC taskFunction = (scope ref const LeastSquaresTask context, uint totalThreads, uint threadId, uint i) @trusted
    {
        context(totalThreads, threadId, i);
    };

    if (tm)
        return optimizeLeastSquaresLM!T(
            lm,
            m,
            x[0 .. n].sliced,
            l[0 .. n].sliced,
            u[0 .. n].sliced,
            (x, y) @trusted => f(fContext, y.length, x.length, x.iterator, y.iterator),
            null,
            (count, scope LeastSquaresTask task) @trusted => tm(tmContext, count, task, taskFunction)
        );
    return optimizeLeastSquaresLM!T(
        lm,
        m,
        x[0 .. n].sliced,
        l[0 .. n].sliced,
        u[0 .. n].sliced,
        (x, y) @trusted => f(fContext, y.length, x.length, x.iterator, y.iterator),
        null,
        null
    );
}

private auto assumePure(T)(T t)
if (isFunctionPointer!T || isDelegate!T)
{
    enum attrs = functionAttributes!T | FunctionAttribute.pure_;
    return cast(SetFunctionAttributes!(T, functionLinkage!T, attrs)) t;
}

// LM algorithm
LMStatus optimizeLMImplGeneric(T)
    (
        scope ref LeastSquaresLM!T lm,
        size_t m,
        Slice!(T*) x,
        Slice!(const(T)*) lower,
        Slice!(const(T)*) upper,
        scope LeastSquaresLM!T.Function f,
        scope LeastSquaresLM!T.Jacobian g,
        scope LeastSquaresThreadManager tm,
    ) @trusted nothrow @nogc pure
{with(lm){
    pragma(inline, false);
    import mir.algorithm.iteration: all;
    import mir.blas;
    import mir.lapack;
    import mir.math.common;
    import mir.math.sum: sum;
    import mir.ndslice.allocation: stdcUninitSlice;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: canonical, diagonal;
    import mir.optim.boxcqp;
    import mir.utility: max;
    import mir.algorithm.iteration;
    import core.stdc.stdio;

    uint n = cast(uint)x.length;

    auto iwork = assumePure(&stdcUninitSlice!(lapackint, 1))([n]);
    auto bwork = assumePure(&stdcUninitSlice!(byte, 1))([n]);
    auto deltaX = assumePure(&stdcUninitSlice!(T, 1))([n]);
    auto Jy = assumePure(&stdcUninitSlice!(T, 1))([n]);
    auto deltaXBase = assumePure(&stdcUninitSlice!(T, 1))([n]);
    auto y = assumePure(&stdcUninitSlice!(T, 1))([m]);
    auto mBuffer = assumePure(&stdcUninitSlice!(T, 1))([m]);
    auto nBuffer = assumePure(&stdcUninitSlice!(T, 1))([n]);
    auto JJ = assumePure(&stdcUninitSlice!(T, 2))([n, n]);
    auto J = assumePure(&stdcUninitSlice!(T, 2))([m, n]);
    auto work = assumePure(&stdcUninitSlice!(T, 1))([boxQPWorkLength(n) + n * 2]);

    scope (exit)
    {
        import mir.internal.memory;
        iwork.ptr.free;
        bwork.ptr.free;
        deltaX.ptr.free;
        Jy.ptr.free;
        deltaXBase.ptr.free;
        y.ptr.free;
        mBuffer.ptr.free;
        nBuffer.ptr.free;
        JJ.ptr.free;
        J.ptr.free;
        work.ptr.free;
    }

    version(LDC) pragma(inline, true);

    if (m == 0 || n == 0 || !x.all!"-a.infinity < a && a < a.infinity")
        return lm.status = LMStatus.badGuess; 
    if (!allLessOrEqual(lower, x) || !allLessOrEqual(x, upper))
        return lm.status = LMStatus.badBounds; 
    if (!(0 <= minStepQuality && minStepQuality < 1))
        return lm.status = LMStatus.badMinStepQuality;
    if (!(0 <= goodStepQuality && goodStepQuality <= 1))
        return lm.status = LMStatus.badGoodStepQuality;
    if (!(minStepQuality < goodStepQuality))
        return lm.status = LMStatus.badStepQuality;
    if (!(1 <= lambdaIncrease && lambdaIncrease <= T.max.sqrt))
        return lm.status = LMStatus.badLambdaParams;
    if (!(T.min_normal.sqrt <= lambdaDecrease && lambdaDecrease <= 1))
        return lm.status = LMStatus.badLambdaParams;

    maxAge = maxAge ? maxAge : g ? 3 : 2 * n;

    if (!tm) tm = delegate(uint count, scope LeastSquaresTask task) pure @nogc nothrow @trusted
    {
        foreach(i; 0 .. count)
            task(1, 0, i);
    };

    f(x, y);
    ++fCalls;
    residual = y.nrm2;


    bool needJacobian = true;
    uint age = maxAge;
    T mu = 1;

    int badPredictions;

    import mir.algorithm.iteration: count;
    import core.stdc.stdio;
    // cast(void) assumePure(&printf)("#### ITERATE ---------\n\n\n");
    lambda = 0;
    iterCt = 0;
    T deltaXBase_nrm2;
    int muFactor = 2;
    do
    {
        if (!allLessOrEqual(x, x))
            return lm.status = LMStatus.numericError;
        if (needJacobian)
        {
            needJacobian = false;
            if (age < maxAge)
            {
                age++;
                auto d = 1 / deltaXBase_nrm2 ^^ 2;
                axpy(-1, y, mBuffer); // -deltaY
                gemv(1, J, deltaXBase, 1, mBuffer); //-(f_new - f_old - J_old*h)
                scal(-d, mBuffer);
                ger(1, mBuffer, deltaXBase, J); //J_new = J_old + u*h'
            }
            else
            {
                age = 0;
                if (g)
                {
                    g(x, J);
                    gCalls += 1;
                }
                else
                {
                    iwork[] = 0;
                    tm(n, (uint totalThreads, uint threadId, uint j)
                        @trusted pure nothrow @nogc
                        {
                            import mir.blas;
                            import mir.math.common;
                            auto idx = totalThreads >= n ? j : threadId;
                            auto p = JJ[idx];
                            if (iwork[idx]++ == 0)
                                copy(x, p);

                            auto save = p[j];
                            auto xmh = save - jacobianEpsilon;
                            auto xph = save + jacobianEpsilon;
                            xmh = fmax(xmh, lower[j]);
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
                                fill(T(0), Jj);
                            }
                        });
                    fCalls += iwork.sum;
                }
            }
            gemv(1, J.transposed, y, 0, Jy);
            // Jy_nrm2 = Jy.nrm2;
        }

        syrk(Uplo.Lower, 1, J.transposed, 0, JJ);

        if (!(lambda >= minLambda))
        {
            lambda = 0.001 * JJ.diagonal[JJ.diagonal.iamax];
            if (!(lambda >= minLambda))
                lambda = 1;
        }
        JJ.diagonal[] += lambda;
        auto l = work[0 .. n];
        auto u = work[n .. n * 2];
        l[] = lower - x;
        u[] = upper - x;
        BoxQPSettings!T settings;
        settings.absTolerance = T.epsilon * 16;
        settings.relTolerance = T.epsilon * 16;
        if (settings.solveBoxQP(JJ.canonical, Jy, l, u, deltaX, false, work[2 * n .. $], iwork, bwork, false) != BoxQPStatus.solved)
        {
            return lm.status = LMStatus.numericError;
        }
        axpy(1, x, deltaX);
        axpy(-1, x, deltaX);
        copy(y, mBuffer);
        gemv(1, J, deltaX, 1, mBuffer); // (J * dx + y) * (J * dx + y)^T
        auto predictedResidual = mBuffer.nrm2;

        if (!(predictedResidual <= residual))
        {
            predictedResidual = residual;
            if (age == 0 && mu == 1)
            {
                cast(void) assumePure(&printf)("#### predictedResidual = %e residual = %e diff = %e\n", predictedResidual, residual, predictedResidual - residual);
                break;
            }
            lambda = fmax(lambdaDecrease * lambda, minLambda);
            mu = 1;
            gConverged = false;
            xConverged = false;
            if (++badPredictions < 8)
                continue;
            else
                break;
        }

        copy(x, nBuffer);
        axpy(1, deltaX, nBuffer);

        applyBounds(nBuffer, lower, upper);

        f(nBuffer, mBuffer);

        ++fCalls;
        auto trialResidual = mBuffer.nrm2;
        if (!(trialResidual <= T.max.sqrt * (1 - T.epsilon)))
            return lm.status = LMStatus.numericError;
        auto improvement = residual ^^ 2 - trialResidual ^^ 2;
        auto predictedImprovement = residual ^^ 2 - predictedResidual ^^ 2;
        auto rho = improvement / predictedImprovement;

        enum maxMu = 8;
        if (rho > minStepQuality && improvement > 0)
        {
            ++iterCt;
            copy(deltaX, deltaXBase);
            deltaXBase_nrm2 = deltaXBase.nrm2;
            if (!(deltaXBase_nrm2 <= T.max.sqrt * (1 - T.epsilon)))
                return lm.status = LMStatus.numericError;
            copy(nBuffer, x);
            swap(y, mBuffer);
            residual = trialResidual;
            needJacobian = true;

            gemv(1, J.transposed, y, 0, nBuffer);
            gConverged = !(nBuffer[nBuffer.iamax].fabs > tolG);
            xConverged = !(deltaXBase_nrm2 > tolX);// fmax(tolX, tolX * x.nrm2));

            if (gConverged && age == 0)
                break;
            if (xConverged)
            {
                if (age == 0) //  && rho > goodStepQuality && muFactor == 1
                    break;
                // cast(void) assumePure(&printf)("#### deltaXBase_nrm2 = %e\n", deltaXBase_nrm2);
                lambda = fmax(lambdaDecrease ^^ 2 * lambda, minLambda);
                muFactor = 1;
                mu = 1;
                gConverged = false;
                xConverged = false;
                age = maxAge;
                continue;
            }

            if (fConverged)
                break;

            if (rho > goodStepQuality)
            {
                lambda = fmax(lambdaDecrease * lambda, minLambda);
                mu = 1;
            }
        }
        else
        {
            if (fConverged)
                break;

            auto newlambda = lambdaIncrease * lambda * mu;
            if (newlambda > maxLambda)
            {
                mu = 1;
                newlambda = lambda + lambda;
            }
            if (newlambda > maxLambda)
            {
                if (age)
                {
                    if (iterCt == 0)
                        assert(0);
                    needJacobian = true;
                    age = maxAge;
                    continue;
                }
                break;
            }
            if (mu < maxMu)
                mu *= muFactor;
            lambda = newlambda;
        }
    }
    while (iterCt < maxIter);

    return lm.status = LMStatus.success;
}}

pragma(inline, false)
void fill(T, SliceKind kind)(T value, Slice!(T*, 1, kind) x)
{
    x[] = value;
}

pragma(inline, false)
bool allLessOrEqual(T)(
    Slice!(const(T)*) a,
    Slice!(const(T)*) b,
    )
{
    import mir.algorithm.iteration: all;
    return all!"a <= b"(a, b);
}

uint normalizeSafety()(uint attrs)
{
    if (attrs & FunctionAttribute.system)
        attrs &= ~FunctionAttribute.safe;
    return attrs;
}

auto trustedAllAttr(T)(scope return T t) @trusted
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
