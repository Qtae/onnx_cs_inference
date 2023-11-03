
using System.Collections.Generic;
using System;
using Microsoft.ML.OnnxRuntime;

namespace ONNX_Inference
{
    public class ADC : ONNXCore
    {
        public ADC() : base() { }

        public ADC(string modelPath, bool bTensorRT, bool bUseCache, string cachePath = "", ulong maxWorkspaceSize = 1ul << 60)
            : base(modelPath, bTensorRT, bUseCache, cachePath, maxWorkspaceSize) { }

        public float[,] RunADCandGetSoftmax(float[,,,] input, int batch)
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

                float[,] res = new float[nImages, nClass];

                return res;
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
