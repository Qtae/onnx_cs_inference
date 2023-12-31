﻿using System;
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
        /* Bump 3d
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
            float[,,] res = bump3dAI.CaculateHeightMap(tmp, batch);
            sw.Stop();
            Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            // sw.Restart();
            // res = bump3dAI.CaculateHeightMap(tmp, batch);
            // sw.Stop();
            // Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            // sw.Restart();
            // res = bump3dAI.CaculateHeightMap(tmp, batch);
            // sw.Stop();
            // Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            // sw.Restart();
            // res = bump3dAI.CaculateHeightMap(tmp, batch);
            // sw.Stop();
            // Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            // sw.Restart();
            // res = bump3dAI.CaculateHeightMap(tmp, batch);
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

        /* ADC
        static void Main(string[] args)
        {
            Console.WriteLine("==========Initializing...==========");
            float[,,,] tmp = new float[10, 200, 200, 3];

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
                    for (int i = 0; i < 200; ++i)
                    {
                        for (int j = 0; j < 200; ++j)
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
            ADC adc = new ADC(modelPath, true, true, cachePath);
            int batch = 2;
            Console.WriteLine("==========Run Inference...==========");
            Stopwatch sw = new Stopwatch();
            sw.Start();
            float[,] res = adc.RunADC(tmp, batch);
            sw.Stop();
            Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            System.Console.WriteLine("==========Success!==========");
            System.Console.ReadKey();
        }
        */
        static void Main(string[] args)
        {
            Console.WriteLine("==========Initializing...==========");
            float[,,,] inspInput = new float[9, 480, 480, 1];
            float[,,,] refInput = new float[9, 480, 480, 1];

            string rootPath = "D:\\QTAE\\CS_ONNX_Inference\\";
            string testDataInspPath = rootPath + "testset_twinnet\\insp\\";
            string testDataRefPath = rootPath + "testset_twinnet\\ref\\";
            string testDataResPath = rootPath + "testset_twinnet\\res\\";

            Console.WriteLine("==========Load Images...==========");
            System.IO.DirectoryInfo testDirInfo = new System.IO.DirectoryInfo(testDataInspPath);
            int imgIdx = 0;
            foreach (System.IO.FileInfo imgFile in testDirInfo.GetFiles())
            {
                if (imgFile.Extension.ToLower().CompareTo(".bmp") == 0)
                {
                    Bitmap img = new Bitmap(imgFile.FullName);
                    for (int i = 0; i < 480; ++i)
                    {
                        for (int j = 0; j < 480; ++j)
                        {
                            inspInput[imgIdx, j, i, 0] = img.GetPixel(i, j).R;
                        }
                    }
                }
                imgIdx++;
            }

            testDirInfo = new System.IO.DirectoryInfo(testDataRefPath);
            imgIdx = 0;
            foreach (System.IO.FileInfo imgFile in testDirInfo.GetFiles())
            {
                if (imgFile.Extension.ToLower().CompareTo(".bmp") == 0)
                {
                    Bitmap img = new Bitmap(imgFile.FullName);
                    for (int i = 0; i < 480; ++i)
                    {
                        for (int j = 0; j < 480; ++j)
                        {
                            refInput[imgIdx, j, i, 0] = img.GetPixel(i, j).R;
                        }
                    }
                }
                imgIdx++;
            }

            Console.WriteLine("==========Load Model...==========");
            string cachePath = rootPath + "models\\";
            string modelPath = cachePath + "480_twin.onnx";
            Twinnet twinnet = new Twinnet(modelPath, true, true, cachePath);
            int batch = 2;
            Console.WriteLine("==========Run Inference...==========");
            Stopwatch sw = new Stopwatch();
            sw.Start();
            float[,,,] res = twinnet.RunTwinnet(inspInput, refInput, batch);
            sw.Stop();

            for (int i = 0; i < res.GetLength(0); ++i)
            {
                Bitmap bitmapImg = new Bitmap(res.GetLength(2), res.GetLength(1));
                for (int y = 0; y < res.GetLength(1); ++y)
                {
                    for (int x = 0; x < res.GetLength(2); ++x)
                    {
                        double denom = Math.Exp(res[i, y, x, 0]) + Math.Exp(res[i, y, x, 1]);
                        double e0 = Math.Exp(res[i, y, x, 0]) / denom;
                        double e1 = Math.Exp(res[i, y, x, 1]) / denom;
                        if (res[i, y, x, 0] > res[i, y, x, 1])
                            bitmapImg.SetPixel(x, y, Color.Red);
                        else
                            bitmapImg.SetPixel(x, y, Color.Black);
                    }
                }
                bitmapImg.Save(testDataResPath + i.ToString() + ".bmp");
            }
            Console.WriteLine("소요 시간: {0}ms", sw.ElapsedMilliseconds);
            System.Console.WriteLine("==========Success!==========");
            System.Console.ReadKey();
        }
    }
}
