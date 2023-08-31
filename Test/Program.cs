using ONNX_Inference;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            ONNXCore onnx = new ONNXCore();
            byte[] tmp = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            bool res = onnx.LoadModel("D:/QTAE/CS_ONNX_Inference/models/stitch.onnx", true, true, "D:/QTAE/CS_ONNX_Inference/models/");
            res = onnx.Run(tmp, tmp, 4);
            if (res)
                System.Console.WriteLine("Success!");
        }
    }
}
