using System;
using System.Collections;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;
using ONNX_Inference;

namespace Test
{
    class Program
    {
        /*
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
            string modelPath = cachePath + "3d_bump_10.onnx";
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

            Console.WriteLine("==========Save result...==========");
            for (int i = 0; i < 1000; ++i)
            {
                Bitmap result = new Bitmap(12, 12);
                for (int x = 0; x < 12; ++x)
                {
                    for (int y = 0; y < 12; ++y)
                    {
                        float pixelFloat = res[i, x, y] * 255.0f / 56.0f;
                        int pixel = (int)pixelFloat;
                        Color col = Color.FromArgb(pixel, pixel, pixel);
                        result.SetPixel(y, x, col);
                    }
                }
                result.Save(rootPath + "test_res\\" + i.ToString() + ".bmp");
            }



            System.Console.WriteLine("==========Success!==========");
            System.Console.ReadKey();
        }
        */


        static void Main(string[] args)
        {
            Console.WriteLine("==========Initializing...==========");
            float[,,,] tmp = new float[30, 32, 32, 3];

            string rootPath = "D:\\QTAE\\CS_ONNX_Inference\\";
            string testDataPath = rootPath + "testset_classification\\";

            Console.WriteLine("==========Load Images...==========");
            System.IO.DirectoryInfo testDirInfo = new System.IO.DirectoryInfo(testDataPath);
            int imgIdx = 0;
            foreach (System.IO.FileInfo imgFile in testDirInfo.GetFiles())
            {
                if (imgFile.Extension.ToLower().CompareTo(".png") == 0)
                {
                    Bitmap img = new Bitmap(imgFile.FullName);
                    for (int i = 0; i < 32; ++i)
                    {
                        for (int j = 0; j < 32; ++j)
                        {
                            tmp[imgIdx, j, i, 0] = img.GetPixel(i, j).R;
                            tmp[imgIdx, j, i, 1] = img.GetPixel(i, j).G;
                            tmp[imgIdx, j, i, 2] = img.GetPixel(i, j).B;
                        }
                    }
                }
                imgIdx++;
            }

            Console.WriteLine("==========Load Model...==========");
            string cachePath = rootPath + "models\\";
            string modelPath = cachePath + "classification.onnx";
            ADC bump3dAI = new ADC(modelPath, true, true, cachePath);
            int batch = 4;
            Console.WriteLine("==========Run Inference...==========");
            Stopwatch sw = new Stopwatch();
            sw.Start();
            float[,] res = bump3dAI.RunADCandGetSoftmax(tmp, batch);
            sw.Stop();
            Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            System.Console.WriteLine("==========Success!==========");
            System.Console.ReadKey();
        }
    }
}
