using System;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;

namespace ONNX_Inference
{
    public class Bump3D : ONNXCore
    {
        public Bump3D() : base() { }

        public Bump3D(string modelPath, bool bTensorRT, bool bUseCache, string cachePath = "", ulong maxWorkspaceSize = 1ul << 60)
            : base(modelPath, bTensorRT, bUseCache, cachePath, maxWorkspaceSize) { }

        public float[,,] GetHeightMap(float[,,,] input, int batch)
        {
            try
            {
                int nImages = input.GetLength(0);
                int nRow = input.GetLength(1);
                int nFOV = input.GetLength(2);
                int nFrames = input.GetLength(3);
                int outputLengthPerImage = nFOV * nFrames;

                float[,,] heightMap = new float[nImages,nFOV,nFrames];

                List<List<int>> inputDims = GetInputDims();
                if (inputDims[0][1] != nRow || inputDims[0][2] != nFOV || inputDims[0][3] != nFrames)
                    throw new Exception("Input dimension is invalid.");

                int[] tensorShape = inputDims[0].ToArray(); //assume that their is only one input operator.
                tensorShape[0] = 1;
                int lengthPerImage = 1;
                foreach (int var in tensorShape) lengthPerImage *= var;
                tensorShape[0] = batch;
                int lengthPerBatch = lengthPerImage * batch;
                int outputLengthPerBatch = outputLengthPerImage * batch;

                // Parallel.For(0, nImages / batch, batchIdx =>
                // {
                //     float[] batchInput = new float[lengthPerBatch];
                //     Buffer.BlockCopy(input, batchIdx * lengthPerBatch * 4, batchInput, 0, lengthPerBatch * 4);
                //     Memory<float> inputMem = new Memory<float>(batchInput);
                //     DenseTensor<float> inputTensor = new DenseTensor<float>(inputMem, tensorShape);
                // 
                //     int batchStart = batchIdx * batch;
                //     int batchEnd = (batchIdx + 1) * batch;
                // 
                //     NamedOnnxValue inputNamedOnnxValue
                //         = NamedOnnxValue.CreateFromTensor(GetInputNames()[0], inputTensor);
                //     List<NamedOnnxValue> inputs = new List<NamedOnnxValue>{ inputNamedOnnxValue };
                //     IReadOnlyCollection<string> outputNames = new List<string> { GetOutputNames()[0] };
                //     IDisposableReadOnlyCollection<DisposableNamedOnnxValue> res = Run(inputs, outputNames);
                // 
                //     Buffer.BlockCopy(
                //         res.ToArray()[0].AsTensor<float>().ToArray(),
                //         0,
                //         heightMap,
                //         batchIdx * outputLengthPerBatch * 4,
                //         outputLengthPerBatch * 4
                //         );
                // });

                for (int batchIdx = 0; batchIdx < nImages / batch; ++batchIdx)
                {
                    float[] batchInput = new float[lengthPerBatch];
                    Buffer.BlockCopy(input, batchIdx * lengthPerBatch * 4, batchInput, 0, lengthPerBatch * 4);
                    Memory<float> inputMem = new Memory<float>(batchInput);
                    DenseTensor<float> inputTensor = new DenseTensor<float>(inputMem, tensorShape);

                    int batchStart = batchIdx * batch;
                    int batchEnd = (batchIdx + 1) * batch;
                
                    NamedOnnxValue inputNamedOnnxValue
                        = NamedOnnxValue.CreateFromTensor(GetInputNames()[0], inputTensor);
                    List<NamedOnnxValue> inputs = new List<NamedOnnxValue> { inputNamedOnnxValue };
                    IReadOnlyCollection<string> outputNames = new List<string> { GetOutputNames()[0] };
                    IDisposableReadOnlyCollection<DisposableNamedOnnxValue> res = Run(inputs, outputNames);

                    Buffer.BlockCopy(
                        res.ToArray()[0].AsTensor<float>().ToArray(),
                        0,
                        heightMap,
                        batchIdx * outputLengthPerBatch * 4,
                        outputLengthPerBatch * 4
                        );
                }

                int residue = nImages % batch;
                tensorShape[0] = residue;
                int lengthOfResidue = lengthPerImage * residue;
                int outputLengthOfResidue = outputLengthPerImage * residue;

                if (residue != 0)
                {
                    int batchStart = nImages - residue;

                    float[] batchInput = new float[lengthOfResidue];
                    Buffer.BlockCopy(input, batchStart * lengthPerImage * 4, batchInput, 0, lengthOfResidue * 4);
                    Memory<float> inputMem = new Memory<float>(batchInput);
                    DenseTensor<float> inputTensor = new DenseTensor<float>(inputMem, tensorShape);

                    NamedOnnxValue inputNamedOnnxValue
                        = NamedOnnxValue.CreateFromTensor(GetInputNames()[0], inputTensor);
                    List<NamedOnnxValue> inputs = new List<NamedOnnxValue> { inputNamedOnnxValue };
                    IReadOnlyCollection<string> outputNames = new List<string> { GetOutputNames()[0] };
                    IDisposableReadOnlyCollection<DisposableNamedOnnxValue> res = Run(inputs, outputNames);

                    Buffer.BlockCopy(
                        res.ToArray()[0].AsTensor<float>().ToArray(),
                        0,
                        heightMap,
                        batchStart * outputLengthPerImage * 4,
                        outputLengthOfResidue * 4
                        );
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
