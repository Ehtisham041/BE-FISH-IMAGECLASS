import { asyncHandler } from "../utils/asynchandler.js";
import {User} from "../models/user.model.js";
import {ApiError} from "../utils/ApiError.js";
import{ApiResponse} from "../utils/ApiResponse.js";
import{uploadOnCloudinary} from "../utils/cloudinary.js";
import jwt from "jsonwebtoken";

const generateAccessTokenAndRefreshToken =async(userid) => 
  {
  try {
     const user = await User.findById(userid);
      if (!user) {
       throw new ApiError(404, "User not found during token generation");
      }

    const accessToken=user.generateAccessToken();
    const refreshToken=user.generateRefreshToken();
    user.refreshToken = refreshToken;
    await user.save({validateBeforeSave: false});
    return {accessToken, refreshToken};

  } catch (error) {
    console.error("Error generating tokens:", error)

    throw new ApiError(500, "Error generating tokens");
    console.error("Error generating tokens:", error);
    throw new ApiError(500, "Failed to generate access token and refresh token");  
  }
}

export const registerUser = asyncHandler(async(req,res,next)=>
  {
    //frist we need to get the data from the request body
    const {username, email, fullname ,password} = req.body;
    console.log({ username, email, fullname, password });


  if ([username, email, fullname, password].some((field) => field?.trim() === "")) 
  {
    throw new ApiError(400, "All fields are required");
    
  }
    //check if the user already exists
    const userExists = await User.findOne({
      $or: [{username}, {email}]
    });
    if(userExists)
    {
      throw new ApiError(409, "User already exists");
    }

   // console.log(req.file);
    const avatar = req.file?.path;
 
   //console.log(avatar);

   if (!avatar) {
    throw new ApiError(400, "Avatar is required");
    
   }
  //upload it on cloudinary 
  const response = await uploadOnCloudinary( avatar);
  if (!response)
    {
    throw new ApiError(500, "Failed to upload avatar on cloudianry");
    }
    const user = await User.create({
      username,
      email,
      fullname, 
      password,
      avatar: response.url
    });

    const createdUser =await  User.findById(user._id).select("-password -refreshToken");
    if (!createdUser) {
      throw new ApiError(500, "Something went wrong Failed to create user");
      
    }
    return res.status(201)
    .json(new ApiResponse(200, createdUser,"user register succesfully "));
   
  }

);


export const loginUser = asyncHandler(async(req,res,next)=>{

  //get the data from the request body
  //check if user email and password are provided
  //check if the user exists
  //check if the password is correct
  //generate access token
  //generate refresh token
  //send the access token and refresh token in the response in cookies 
  //save the refresh token in the database
  //send the response
  
  const {username, email, password} = req.body;
  if (![username, email, password].every((field) => field?.trim() !== "")) {
    throw new ApiError(400, "All fields are required");
  }
  //check if user exists
  const user = await User.findOne({$or:[{username},{email}]});
  
  if (!user) {
    throw new ApiError(404, "User not found");
  }
  //check if password is correct
  const isPasswordCorrect = await user.isPasswordCorrect(password);
  if (!isPasswordCorrect) {
    throw new ApiError(401, "Invalid credentials");
  }
  //generate access token and refresh token
  const {accessToken, refreshToken} = await generateAccessTokenAndRefreshToken(user._id);
  
  //ab cookies bhejne ha aur cookies se pahley options set karna hoga
  const options = {
    httpOnly: true,//iska mtlb ye ha ke ye cookie sirf server se communicate karega
                   //aur client side se nahi aur iska faida ye ha ke agar koi hacker
                   //  client side se cookie ko access karna chata ha to wo nahi kar sakta
    secure : true,
    //ye option tab use hota ha jab hum https use karte ha
    

  }
  return res.status(200)
  .cookie("accessToken", accessToken, options)
  .cookie("refreshToken", refreshToken, options)
  .json
  (
    new ApiResponse(200, 
      {accessToken, refreshToken},//ye humne response me bheja ha although ye cookies me bhejna tha maagr humne response me bheja ha 
       "User logged in successfully")
  );



});

export const logoutUser = asyncHandler(async(req,res,next)=>
{
    
const user = req.user._id;
await User.findByIdAndUpdate(user, 
  {
  $set: 
  {
    refreshToken: undefined
  } 
  }, 
  {
    new: true
  });


  const options = {
    httpOnly: true,
    secure : true,

  }

  return res.status(200)
  .clearCookie("accessToken", options)
  .clearCookie("refreshToken", options)
  .json(new ApiResponse(200, {}, "User logged out successfully"));

});

