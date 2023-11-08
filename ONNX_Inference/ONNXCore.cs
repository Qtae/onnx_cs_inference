using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ONNX_Inference
{
	public class ONNXCore
	{
		private InferenceSession inferenceSession;
		protected IReadOnlyDictionary<string, NodeMetadata> inputMetaData;
		protected IReadOnlyDictionary<string, NodeMetadata> outputMetaData;
		protected bool bIsModelLoaded = false;
		protected bool bIs16bitModel = false;

		public ONNXCore()
        {
            try
            {
                OrtEnv.Instance();
            }
            catch (OnnxRuntimeException ex)
            {
                System.Console.WriteLine("Error in ONNXCore() : " + ex.Message);
                throw;
            }
            catch (Exception ex)
            {
                System.Console.WriteLine("Error in ONNXCore() : " + ex.Message);
                throw;
            }
        }

		public ONNXCore(string modelPath, bool bTensorRT, bool bUseCache,
            string cachePath = "", ulong maxWorkspaceSize = 1ul << 60)
		{
			try
            {
                OrtEnv.Instance();
                LoadModel(modelPath, bTensorRT, bUseCache, cachePath, maxWorkspaceSize);
            }
            catch (OnnxRuntimeException ex)
            {
                System.Console.WriteLine("Error in ONNXCore() : " + ex.Message);
                throw;
            }
            catch (Exception ex)
            {
                System.Console.WriteLine("Error in ONNXCore() : " + ex.Message);
                throw;
            }
        }

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
			try
            {
                List<List<int>> inputDims = new List<List<int>>();
                IEnumerator<NodeMetadata> nodeMataDataEnum = InputMetaData.Values.GetEnumerator();
                while (nodeMataDataEnum.MoveNext() == true)
                {
                    List<int> tmp = new List<int>();
                    int[] arr = nodeMataDataEnum.Current.Dimensions;
                    foreach (int val in arr) tmp.Add(val);
                    inputDims.Add(tmp);
                }
                return inputDims;
            }
            catch (OnnxRuntimeException ex)
            {
                System.Console.WriteLine("Error in GetInputDims() : " + ex.Message);
                throw;
            }
            catch (Exception ex)
            {
                System.Console.WriteLine("Error in GetInputDims() : " + ex.Message);
                throw;
            }
        }
		public List<List<int>> GetOutputDims()
        {
			try
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
            catch (OnnxRuntimeException ex)
            {
                System.Console.WriteLine("Error in GetOutputDims() : " + ex.Message);
                throw;
            }
            catch (Exception ex)
            {
                System.Console.WriteLine("Error in GetOutputDims() : " + ex.Message);
                throw;
            }
        }
        public List<string> GetInputNames()
        {
            try
            {
                List<string> inputNames = new List<string>();
                foreach (var key in InputMetaData.Keys) inputNames.Add(key);
                return inputNames;
            }
            catch (OnnxRuntimeException ex)
            {
                System.Console.WriteLine("Error in GetInputNames() : " + ex.Message);
                throw;
            }
            catch (Exception ex)
            {
                System.Console.WriteLine("Error in GetInputNames() : " + ex.Message);
                throw;
            }
        }
        public List<string> GetOutputNames()
        {
            try
            {
                List<string> outputNames = new List<string>();
                foreach (var key in OutputMetaData.Keys) outputNames.Add(key);
                return outputNames;
            }
            catch (OnnxRuntimeException ex)
            {
                System.Console.WriteLine("Error in GetOutputNames() : " + ex.Message);
                throw;
            }
            catch (Exception ex)
            {
                System.Console.WriteLine("Error in GetOutputNames() : " + ex.Message);
                throw;
            }
        }

        public bool LoadModel(string modelPath, bool bTensorRT, bool bUseCache,
			string cachePath = "", ulong maxWorkspaceSize = 1ul << 60)
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
						["trt_max_workspace_size"] = maxWorkspaceSize.ToString(),
						["trt_engine_cache_enable"] = bUseCache ? "true" : "false",
						["trt_engine_cache_path"] = cachePath == "" ? modelPath : cachePath
					};
					trtOptions.UpdateOptions(providerOptionsDict);
					sessionOptions.AppendExecutionProvider_Tensorrt(trtOptions);
					trtOptions.Close();
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
				System.Console.WriteLine("Error in LoadModel() : " + ex.Message);
				throw;
            }
            catch (Exception ex)
            {
                System.Console.WriteLine("Error in LoadModel() : " + ex.Message);
                throw;
            }
        }

		protected IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(
			IReadOnlyCollection<NamedOnnxValue> namedInputs, IReadOnlyCollection<string> outputNames,
			RunOptions runOptions = null)
        {
			try
            {
				if (!bIsModelLoaded)
				{
					System.Console.WriteLine("Model not loaded!");
					return null;
				}

                if (runOptions == null)
                {
                    runOptions = new RunOptions();
                }

				IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result
					= inferenceSession.Run(namedInputs, outputNames, runOptions);

				return result;
            }
			catch (OnnxRuntimeException ex)
			{
				System.Console.WriteLine("Error in Run() : " + ex.Message);
				throw;
            }
            catch (Exception ex)
            {
                System.Console.WriteLine("Error in Run() : " + ex.Message);
                throw;
            }
        }
	}
}
