using System;
using System.Collections;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
using ONNX_Inference;

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
            Console.WriteLine("==========Initializing...==========");
            float[,,,] tmp = new float[1000, 56, 12, 12];

            string rootPath = "D:\\QTAE\\CS_ONNX_Inference\\";
            string testDataPath = rootPath + "testset\\";

            Console.WriteLine("==========Load Images...==========");
            System.IO.DirectoryInfo testRootDirInfo = new System.IO.DirectoryInfo(testDataPath);
            int setIdx = 0;
            foreach (System.IO.DirectoryInfo subDir in testRootDirInfo.GetDirectories())
            {
                string testDataSubDirPath = subDir.FullName;
                string convertedDataPath = testDataSubDirPath + "\\converted";
                System.IO.DirectoryInfo convertedDirInfo = new System.IO.DirectoryInfo(convertedDataPath);
                int imgIdx = 0;
                foreach (System.IO.FileInfo imgFile in convertedDirInfo.GetFiles())
                {
                    if (imgFile.Extension.ToLower().CompareTo(".png") == 0)
                    {
                        Bitmap img = new Bitmap(imgFile.FullName);
                        for(int i = 0; i < 12; ++i)
                        {
                            for (int j = 0; j < 56; ++j)
                            {
                                tmp[setIdx, j, i, imgIdx] = img.GetPixel(i, j).R;
                            }
                        }
                    }
                    imgIdx++;
                }
                setIdx++;
                if (setIdx == 1000) break;
            }

            Console.WriteLine("==========Load Model...==========");
            string cachePath = rootPath + "models\\";
            string modelPath = cachePath + "3d_bump.onnx";
            Bump3D bump3dAI = new Bump3D(modelPath, true, true, cachePath);
            int batch = 32;
            Console.WriteLine("==========Run Inference...==========");
            Stopwatch sw = new Stopwatch();
            sw.Start();
            float[,,] res = bump3dAI.GetHeightMap(tmp, batch);
            sw.Stop();
            Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            // sw.Restart();
            // res = bump3dAI.GetHeightMap(tmp, batch);
            // sw.Stop();
            // Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            // sw.Restart();
            // res = bump3dAI.GetHeightMap(tmp, batch);
            // sw.Stop();
            // Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            // sw.Restart();
            // res = bump3dAI.GetHeightMap(tmp, batch);
            // sw.Stop();
            // Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            // sw.Restart();
            // res = bump3dAI.GetHeightMap(tmp, batch);
            // sw.Stop();
            // Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            System.Console.WriteLine("==========Success!==========");
            System.Console.ReadKey();
        }
    }
}
