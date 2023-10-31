using System;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;

namespace ONNX_Inference
{
    public class Bump3D : ONNXCore
    {
        public Bump3D() : base() { }

        public Bump3D(string modelPath, bool bTensorRT, bool bUseCache, string cachePath = "", ulong maxWorkspaceSize = 1ul << 60)
            : base(modelPath, bTensorRT, bUseCache, cachePath, maxWorkspaceSize) { }

        public byte[] GetHeightMap(byte[,,,] input, int batch)
        {
            try
            {
                int nImages = input.GetLength(0);
                int nRow = input.GetLength(1);
                int mFOV = input.GetLength(2);
                int nFrames = input.GetLength(3);

                byte[] heightMap = new byte[nImages * mFOV * nFrames];

                List<List<int>> inputDims = GetInputDims();
                if (inputDims[0][1] != nRow || inputDims[0][2] != mFOV || inputDims[0][3] != nFrames)
                    throw new Exception("Input dimension is invalid.");

                int[] tensorShape = inputDims[0].ToArray(); //assume that their is only one input operator.
                tensorShape[0] = batch;
                DenseTensor<float> inputTensor = new DenseTensor<float>(tensorShape);

                for (int batchIdx = 0; batchIdx < nImages / batch + (nImages % batch == 0 ? 0 : 1) ; ++batchIdx)
                {
                    int batchStart = batchIdx * batch;
                    int batchEnd = (batchIdx + 1) * batch;
                    Parallel.For(batchStart, batchEnd, imgIdx =>
                    {
                        for (int rowIdx = 0; rowIdx < nRow; ++rowIdx)
                        {
                            for (int fovIdx = 0; fovIdx < mFOV; ++fovIdx)
                            {
                                for (int frameIdx = 0; frameIdx < nFrames; ++frameIdx)
                                {
                                    inputTensor[imgIdx - batchStart, rowIdx, fovIdx, frameIdx, 0] =
                                        input[imgIdx, rowIdx, fovIdx, frameIdx];
                                }
                            }
                        }
                    });

                    NamedOnnxValue inputNamedOnnxValue
                        = NamedOnnxValue.CreateFromTensor(GetInputNames()[0], inputTensor);
                    List<NamedOnnxValue> inputs = new List<NamedOnnxValue>{ inputNamedOnnxValue };
                    IReadOnlyCollection<string> outputNames = new List<string> { GetOutputNames()[0] };
                    IDisposableReadOnlyCollection<DisposableNamedOnnxValue> res = Run(inputs, outputNames);
                }

                return heightMap;
            }

            catch (OnnxRuntimeException ex)
            {
                System.Console.WriteLine("Error in GetHeightMap() :");
                System.Console.WriteLine(ex.Message);
                throw;
            }

            catch (Exception ex)
            {
                System.Console.WriteLine("Error in GetHeightMap() :");
                System.Console.WriteLine(ex.Message);
                throw;
            }
        }
    }
}
