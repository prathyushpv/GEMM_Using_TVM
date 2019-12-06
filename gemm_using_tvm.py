import tvm
import numpy
import timeit

M = 1024
K = 1024
N = 1024

dtype = "float32"

target = 'llvm -mcpu=core-avx2'
ctx = tvm.context(target, 0)
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), ctx)
np_repeat = 10
np_runing_time = timeit.timeit(setup='import numpy\n'
                                     'M = ' + str(M) + '\n'
                                     'K = ' + str(K) + '\n'
                                     'N = ' + str(N) + '\n'
                                     'dtype = "float32"\n'
                                     'a = numpy.random.rand(M, K).astype(dtype)\n'
                                     'b = numpy.random.rand(K, N).astype(dtype)\n',
                               stmt='answer = numpy.dot(a, b)',
                               number=np_repeat)
print("Numpy : %f" % (np_runing_time / np_repeat))

answer = numpy.dot(a.asnumpy(), b.asnumpy())

k = tvm.reduce_axis((0, K), 'k')
MatrixA = tvm.placeholder((M, K), name='MatrixA')
MatrixB = tvm.placeholder((K, N), name='MatrixB')
CMatrix = tvm.compute(
           (M, N),
           lambda x, y: tvm.sum(MatrixA[x, k] * MatrixB[k, y], axis=k),
           name='CMatrix')

s = tvm.create_schedule(CMatrix.op)
f = tvm.build(s, [MatrixA, MatrixB, CMatrix], target=target, name='mmult')
c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
f(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
eval = f.time_evaluator(f.entry_name, ctx, number=1)
print('Without optimizations: %f' % eval(a, b, c).mean)



bn = 32
packedMatrixB = tvm.compute((N / bn, K, bn), lambda x, y, z: MatrixB[y, x * bn + z], name='packedMatrixB')
CMatrix = tvm.compute((M, N),
                lambda x, y: tvm.sum(MatrixA[x, k] * packedMatrixB[y // bn, k, tvm.indexmod(y, bn)], axis=k),
                name = 'CMatrix')
s = tvm.create_schedule(CMatrix.op)
newCMatrix = s.cache_write(CMatrix, 'global')
xo, yo, xi, yi = s[CMatrix].tile(CMatrix.op.axis[0], CMatrix.op.axis[1], bn, bn)
s[newCMatrix].compute_at(s[CMatrix], yo)
xc, yc = s[newCMatrix].op.axis
k, = s[newCMatrix].op.reduce_axis
ko, ki = s[newCMatrix].split(k, factor=4)
s[newCMatrix].reorder(ko, xc, ki, yc)
s[newCMatrix].unroll(ki)
s[newCMatrix].vectorize(yc)
#s[CMatrix].parallel(xo)

x, y, z = s[packedMatrixB].op.axis
s[packedMatrixB].vectorize(z)
s[packedMatrixB].parallel(x)
f = tvm.build(s, [MatrixA, MatrixB, CMatrix], target=target, name = 'mmult')
c = tvm.nd.array(numpy.zeros((M, N), dtype = dtype), ctx)
f(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
eval = f.time_evaluator(f.entry_name, ctx, number=50)
opt6_time = eval(a, b, c).mean
print('With tvm optimizations: %f' % opt6_time)
