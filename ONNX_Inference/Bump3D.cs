
namespace ONNX_Inference
{
    public class Bump3D : ONNXCore
    {
        public Bump3D() : base() { }

        public Bump3D(string modelPath, bool bTensorRT, bool bUseCache, string cachePath = "", ulong maxWorkspaceSize = 1ul << 60)
            : base(modelPath, bTensorRT, bUseCache, cachePath, maxWorkspaceSize) { }

        public byte[] GetHeightMap(byte[] inputImageArr)
        {
            byte[] tmp = { };
            base.Run(inputImageArr, tmp, 4);
            return tmp;
        }
    }
}
