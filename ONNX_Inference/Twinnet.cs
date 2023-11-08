
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Threading.Tasks;
using System;
using System.Linq;

namespace ONNX_Inference
{
    public class Twinnet : ONNXCore
    {
        public Twinnet() : base() { }

        public Twinnet(string modelPath, bool bTensorRT, bool bUseCache, string cachePath = "", ulong maxWorkspaceSize = 1ul << 60)
            : base(modelPath, bTensorRT, bUseCache, cachePath, maxWorkspaceSize) { }

        public float[,,,] RunTwinnet(float[,,,] inspInput, float[,,,] refInput, int batch)
        {
            try
            {
                int nImages = inspInput.GetLength(0);
                int nHeight = inspInput.GetLength(1);
                int nWidth = inspInput.GetLength(2);
                int nChannel = inspInput.GetLength(3);

                if (nImages != inspInput.GetLength(0) ||
                    nHeight != inspInput.GetLength(1) ||
                    nWidth != inspInput.GetLength(2) ||
                    nChannel != inspInput.GetLength(3))
                {
                    throw new Exception("Twinnet inspection input dimension and reference input dimension are different.");
                }

                int outputLengthPerImage = nHeight * nWidth * 2;

                float[,,,] twinnetRes = new float[nImages, nHeight, nWidth, 2];

                List<List<int>> inputDims = GetInputDims();
                if (inputDims[0][1] != nHeight || inputDims[0][2] != nWidth || inputDims[0][3] != nChannel)
                    throw new Exception("Input dimension of twinnet model and image dimension are different.");

                int[] tensorShape = inputDims[0].ToArray(); //assume that their is only one input operator.
                tensorShape[0] = 1;
                int lengthPerImage = 1;
                foreach (int var in tensorShape) lengthPerImage *= var;
                tensorShape[0] = batch;
                int lengthPerBatch = lengthPerImage * batch;
                int outputLengthPerBatch = outputLengthPerImage * batch;

                Parallel.For(0, nImages / batch, batchIdx =>
                {
                    float[] batchInputInsp = new float[lengthPerBatch];
                    float[] batchInputRef = new float[lengthPerBatch];
                    Buffer.BlockCopy(inspInput, batchIdx * lengthPerBatch * 4, batchInputInsp, 0, lengthPerBatch * 4);
                    Buffer.BlockCopy(refInput, batchIdx * lengthPerBatch * 4, batchInputRef, 0, lengthPerBatch * 4);
                    Memory<float> inputMemInsp = new Memory<float>(batchInputInsp);
                    Memory<float> inputMemRef = new Memory<float>(batchInputRef);
                    DenseTensor<float> inputTensorInsp = new DenseTensor<float>(inputMemInsp, tensorShape);
                    DenseTensor<float> inputTensorRef = new DenseTensor<float>(inputMemRef, tensorShape);

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
                        twinnetRes,
                        batchIdx * outputLengthPerBatch * 4,
                        outputLengthPerBatch * 4
                        );
                });

                int residue = nImages % batch;
                tensorShape[0] = residue;
                int lengthOfResidue = lengthPerImage * residue;
                int outputLengthOfResidue = outputLengthPerImage * residue;

                if (residue != 0)
                {
                    int batchStart = nImages - residue;

                    float[] batchInput = new float[lengthOfResidue];
                    Buffer.BlockCopy(inspInput, batchStart * lengthPerImage * 4, batchInput, 0, lengthOfResidue * 4);
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
                        twinnetRes,
                        batchStart * outputLengthPerImage * 4,
                        outputLengthOfResidue * 4
                        );
                }

                return twinnetRes;
            }

            catch (OnnxRuntimeException ex)
            {
                System.Console.WriteLine("Error in RunTwinnet() : " + ex.Message);
                throw;
            }

            catch (Exception ex)
            {
                System.Console.WriteLine("Error in RunTwinnet() : " + ex.Message);
                throw;
            }
        }
    }
}
