using System;
using System.Collections;
using ONNX_Inference;
//using SixLabors.ImageSharp;

namespace Test
{
    class Program
    {
        static IEnumerator Test()
        {
            Console.WriteLine("1");
            Console.WriteLine("2");
            Console.WriteLine("3");
            yield return null;
        }
        static void Main(string[] args)
        {
            string modelPath = "D:\\QTAE\\CS_ONNX_Inference\\models\\stitch.onnx";
            string cachePath = "D:\\QTAE\\CS_ONNX_Inference\\models\\";
            Bump3D bump3dAI = new Bump3D(modelPath, true, true, cachePath);
            byte[] tmp = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            byte[] res = bump3dAI.GetHeightMap(tmp);
            //bool res = onnx.LoadModel(modelPath, true, true, cachePath);
            System.Console.WriteLine("Success!");
            //IEnumerator test = Test();
            //test.MoveNext();
        }
    }
}
