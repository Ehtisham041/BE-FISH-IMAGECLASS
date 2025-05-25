import { asyncHandler } from "../utils/asynchandler.js";
import { Fish } from "../models/fish.model.js";
import { ApiResponse } from "../utils/ApiResponse.js";
import { ApiError } from "../utils/ApiError.js";
import { uploadOnCloudinary } from "../utils/cloudinary.js";
import axios from "axios";

export const uploadFishAndPredict = asyncHandler(async (req, res) => {
  const image = req.file?.path;
  if (!image) throw new ApiError(400, "Fish image is required");

  const uploadedImage = await uploadOnCloudinary( image );
  if (!uploadedImage){ 
    throw new ApiError(500, "Failed to upload fish image");
  }
  console.log("Calling FastAPI /predict with image_url:", uploadedImage.url)

  // 🔁 Call FastAPI Microservice
  const { data } = await axios.post(
    "http://localhost:8001/predict",
    { image_url: uploadedImage.url },
    {
      headers: {
        Authorization: `Bearer ${req.cookies.accessToken}`,
        "Content-Type": "application/json"
      }
    },
    

  );

const savedFish = await Fish.create({
  species: data.species,             // ✅ matches schema
  confidence: data.confidence,
  imageUrl: uploadedImage.url,
  uploadedBy: req.user._id           // ✅ matches schema
});



  res.status(201).json(new ApiResponse(200, savedFish, "Prediction complete and saved"));
});

export const getUserFishPredictions = asyncHandler(async (req, res) => {
  const fishList = await Fish.find({ uploadedBy: req.user._id }).sort({ createdAt: -1 });
  res.status(200).json(new ApiResponse(200, fishList));
  
});
