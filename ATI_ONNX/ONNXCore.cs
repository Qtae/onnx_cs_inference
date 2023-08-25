using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;


namespace ATI_ONNX
{
	public class ONNXCore
	{
		private string mModelPath;
		private InferenceSession mSession;
		private bool mbIsModelLoaded = false;

		public bool LoadModel(string modelPath, bool bTensorRT, bool bUseCache, string cachePath = "")
		{
			mbIsModelLoaded = false;
			mModelPath = modelPath;
			SessionOptions sessionOptions = new SessionOptions();
			if (bTensorRT)
			{
				OrtTensorRTProviderOptions trtOptions = new OrtTensorRTProviderOptions();
				var providerOptionsDict = new Dictionary<string, string>
				{
					["device_id"] = "0",
					["trt_fp16_enable"] = "true",
					["trt_engine_cache_enable"] = bUseCache? "true" : "false",
					["trt_engine_cache_path"] = cachePath == "" ? modelPath : cachePath
				};
				trtOptions.UpdateOptions(providerOptionsDict);
				sessionOptions.AppendExecutionProvider_Tensorrt(trtOptions);
				sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
			}
            else return false;

			mSession = new InferenceSession(mModelPath, sessionOptions);

			return true;
		}
	}
}
