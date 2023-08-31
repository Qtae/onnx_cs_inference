using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;

namespace ONNX_INF_CORE
{
	public class ONNXCore
	{
		private InferenceSession inferenceSession;
		protected IReadOnlyDictionary<string, NodeMetadata> inputMetaData;
		protected IReadOnlyDictionary<string, NodeMetadata> outputMetaData;
		protected bool bIsModelLoaded = false;
		protected bool bIs16bitModel = false;

		public bool IsModelLoaded
		{
			get { return bIsModelLoaded; }
		}

		public bool Is16bitModel
		{
			get { return bIs16bitModel; }
		}

		public IReadOnlyDictionary<string, NodeMetadata> InputMetaData
		{
			get { return inputMetaData; }
		}

		public IReadOnlyDictionary<string, NodeMetadata> OutputMetaData
		{
			get { return outputMetaData; }
		}
		public List<List<int>> GetInputDims()
        {
			List<List<int>> inputDims = new List<List<int>>();
			IEnumerator<NodeMetadata> nodeMataDataEnum = InputMetaData.Values.GetEnumerator();
			while (nodeMataDataEnum.MoveNext() == true)
            {
				List<int> tmp = new List<int>();
				int[] arr =  nodeMataDataEnum.Current.Dimensions;
				foreach (int val in arr) tmp.Add(val);
				inputDims.Add(tmp);
			}
			return inputDims;
        }
		public List<List<int>> GetOutputDims()
        {
			List<List<int>> outputDims = new List<List<int>>();
			IEnumerator<NodeMetadata> nodeMataDataEnum = OutputMetaData.Values.GetEnumerator();
			while (nodeMataDataEnum.MoveNext() == true)
			{
				List<int> tmp = new List<int>();
				int[] arr = nodeMataDataEnum.Current.Dimensions;
				foreach (int val in arr) tmp.Add(val);
				outputDims.Add(tmp);
			}
			return outputDims;
		}

		protected bool LoadModel(string modelPath, bool bTensorRT, bool bUseCache, string cachePath = "")
        {
            try
            {
				bIsModelLoaded = false;
				SessionOptions sessionOptions = new SessionOptions();
				if (bTensorRT)
				{
					OrtTensorRTProviderOptions trtOptions = new OrtTensorRTProviderOptions();
					Dictionary<string, string> providerOptionsDict = new Dictionary<string, string>()
					{
						["device_id"] = "0",
						["trt_fp16_enable"] = "true",
						["trt_engine_cache_enable"] = bUseCache ? "true" : "false",
						["trt_engine_cache_path"] = cachePath == "" ? modelPath : cachePath
					};
					trtOptions.UpdateOptions(providerOptionsDict);
					sessionOptions.AppendExecutionProvider_Tensorrt(trtOptions);
					sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
				}
				else return false;

				inferenceSession = new InferenceSession(modelPath, sessionOptions);

				inputMetaData = inferenceSession.InputMetadata;
				outputMetaData = inferenceSession.OutputMetadata;

				//mType 32bit float : "Single", 16bit Float : "Float16"
				IEnumerator<NodeMetadata> metaEnumerator = inputMetaData.Values.GetEnumerator();
				metaEnumerator.MoveNext();
				string valType = metaEnumerator.Current.ElementType.Name;
				bIs16bitModel = valType == "Float16";

				bIsModelLoaded = true;

				return true;
			}

            catch (OnnxRuntimeException ex)
			{
				System.Console.WriteLine("Error in LoadModel() :");
				System.Console.WriteLine(ex.Message);
				throw;
			}
            
		}

		protected bool Run(byte[] InputImageArray, byte[] outputImageArray)
        {
			try
            {
				if (!bIsModelLoaded)
				{
					System.Console.WriteLine("Model not loaded!");
					return false;
				}

				float[] InputImgFloatArray = new float[InputImageArray.Length];
				Parallel.For(0, InputImageArray.Length, i => InputImgFloatArray[i] = ((float)InputImageArray[i] / 255.0f));

				int inputDimCheckVal = 1;
				List<List<int>> inputDims = GetInputDims();

				foreach (List<int> inputDim in inputDims)
                {
					foreach(int val in inputDim)
                    {
						inputDimCheckVal *= val;
                    }
                }					
				if (InputImageArray.Length != inputDimCheckVal)// compare with length --> 추후 Input check 함수를 따로 만들어야 함.
				{
					System.Console.WriteLine("Model input and image input is not compatible!");
					return false;
                }

				IReadOnlyCollection<NamedOnnxValue> inputs = new List<NamedOnnxValue>();
				IReadOnlyCollection<NamedOnnxValue> outputs = new List<NamedOnnxValue>();

				inferenceSession.Run(inputs, outputs);

				return true;
            }

			catch (OnnxRuntimeException ex)
			{
				System.Console.WriteLine("Error in LoadModel() :");
				System.Console.WriteLine(ex.Message);
				throw;
			}
        }
	}
}
