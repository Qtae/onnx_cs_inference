
namespace ONNX_Inference
{
    public class Twinnet : ONNXCore
    {
        public Twinnet() : base() { }

        public Twinnet(string modelPath, bool bTensorRT, bool bUseCache, string cachePath = "", ulong maxWorkspaceSize = 1ul << 60)
            : base(modelPath, bTensorRT, bUseCache, cachePath, maxWorkspaceSize) { }
    }
}
