import {v2 as cloudinary } from 'cloudinary'
import fs from 'fs'

console.log("Cloudinary config loaded");
import dotenv from 'dotenv';
dotenv.config();
cloudinary.config({

    
    cloud_name:process.env.CLOUDINARY_NAME,
    api_key:process.env.CLOUDINARY_API_KEY,
    api_secret:process.env.CLOUDINARY_API_SECRET
});


const uploadOnCloudinary = async (localFilePath)=>{


    try{

        if(!localFilePath)
        {
            return null;
        }
        //upload file on cloudinary

        const response =await cloudinary.uploader.upload(localFilePath,{
            resource_type:"auto"
        })
        console.log("file is uploaded on clodinary",response.url)
        return response;


    }
    catch(error)
    {
fs.unlinkSync(localFilePath)
return null
    }
}
export {uploadOnCloudinary}

// import fs from 'fs';

// const uploadOnCloudinary = async (localFilePath) => {
//   try {
//     if (!localFilePath) {
//       return null;
//     }
//     // Upload file on cloudinary
//     const response = await cloudinary.uploader.upload(localFilePath, {
//       resource_type: "auto"
//     });
//     console.log("file is uploaded on cloudinary", response.url);

//     // Delete local file after successful upload
//     if (fs.existsSync(localFilePath)) {
//       fs.unlinkSync(localFilePath);
//     }

//     return response;
//   } catch (error) {
//     console.error("Cloudinary upload error:", error);

//     // Delete local file in case of error too, if file exists
//     if (localFilePath && typeof localFilePath === 'string' && fs.existsSync(localFilePath)) {
//       fs.unlinkSync(localFilePath);
//     }

//     return null;
//   }
// }
// export {uploadOnCloudinary}