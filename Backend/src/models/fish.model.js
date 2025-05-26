import mongoose, { Schema } from "mongoose";

const fishSchema = new Schema(
  {
    imageUrl: {
      type: String,
      required: true,
    
    },
    species: {            // renamed from predictedClass
      type: String,
      required: true,
    },
    confidence: {
      type: Number,
      required: false,
    },
    uploadedBy: {         // renamed from user
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    model: { type: String, default: "VitTransformer" }
  },
  { timestamps: true }
);


export const Fish = mongoose.model("Fish", fishSchema);
