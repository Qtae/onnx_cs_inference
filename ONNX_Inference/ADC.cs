using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ONNX_Inference
{
    public class ADC : ONNXCore
    {
        public ADC() : base() { }

        public ADC(string modelPath, bool bTensorRT, bool bUseCache, string cachePath = "", ulong maxWorkspaceSize = 1ul << 60)
            : base(modelPath, bTensorRT, bUseCache, cachePath, maxWorkspaceSize) { }

        public float[,] RunADC(float[,,,] input, int batch)
        {
            try
            {
                int nImages = input.GetLength(0);
                int nHeight = input.GetLength(1);
                int nWidth = input.GetLength(2);
                int nChannel = input.GetLength(3);

                List<List<int>> inputDims = GetInputDims();
                if (inputDims[0][1] != nHeight || inputDims[0][2] != nWidth || inputDims[0][3] != nChannel)
                    throw new Exception("Input dimension is invalid.");

                List<List<int>> outputDims = GetOutputDims();
                int nClass = outputDims[0][1];

                float[,] softmx = new float[nImages, nClass];

                int[] tensorShape = inputDims[0].ToArray(); //assume that their is only one input operator.
                tensorShape[0] = 1;
                int lengthPerImage = 1;
                foreach (int var in tensorShape) lengthPerImage *= var;
                tensorShape[0] = batch;
                int lengthPerBatch = lengthPerImage * batch;
                int outputLengthPerBatch = nClass * batch;

                Parallel.For(0, nImages / batch, batchIdx =>
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
                        softmx,
                        batchIdx * outputLengthPerBatch * 4,
                        outputLengthPerBatch * 4
                        );
                    res.Dispose();
                });

                int residue = nImages % batch;
                tensorShape[0] = residue;
                int lengthOfResidue = lengthPerImage * residue;
                int outputLengthOfResidue = nClass * residue;

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
                        softmx,
                        batchStart * nClass * 4,
                        outputLengthOfResidue * 4
                        );
                    res.Dispose();
                }

                return softmx;
            }
            catch (OnnxRuntimeException ex)
            {
                System.Console.WriteLine("Error in RunADCandGetSoftmax() :");
                System.Console.WriteLine(ex.Message);
                throw;
            }

            catch (Exception ex)
            {
                System.Console.WriteLine("Error in RunADCandGetSoftmax() :");
                System.Console.WriteLine(ex.Message);
                throw;
            }
        }
    }
}
