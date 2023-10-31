
namespace ONNX_Inference
{
    public class ADC : ONNXCore
    {
        public ADC() : base() { }

        public ADC(string modelPath, bool bTensorRT, bool bUseCache, string cachePath = "", ulong maxWorkspaceSize = 1ul << 60)
            : base(modelPath, bTensorRT, bUseCache, cachePath, maxWorkspaceSize) { }
    }
}
