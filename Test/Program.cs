using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ATI_ONNX;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            ONNXCore onnxCore = new ONNXCore();
            onnxCore.LoadModel("./model.onnx", true, true, "");
        }
    }
}
