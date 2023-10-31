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
            string modelPath = "D:\\QTAE\\CS_ONNX_Inference\\models\\3d_bump.onnx";
            string cachePath = "D:\\QTAE\\CS_ONNX_Inference\\models\\";
            Bump3D bump3dAI = new Bump3D(modelPath, true, true, cachePath);
            byte[,,,] tmp = new byte[1000, 56, 12, 12];
            for (int i = 0; i < tmp.GetLength(0); i++)
            {
                for (int j = 0; j < tmp.GetLength(1); j++)
                {
                    for (int k = 0; k < tmp.GetLength(2); k++)
                    {
                        for (int n = 0; n < tmp.GetLength(3); n++)
                            tmp[i, j, k, n] = 0;
                    }
                }
            }
            byte[] res = bump3dAI.GetHeightMap(tmp, 4);
            //bool res = onnx.LoadModel(modelPath, true, true, cachePath);
            System.Console.WriteLine("Success!");
            //IEnumerator test = Test();
            //test.MoveNext();
            System.Console.ReadKey();
        }
    }
}
