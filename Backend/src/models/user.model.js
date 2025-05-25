import mongoose,{Schema} from "mongoose";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";

const userSchema = new Schema({

username :
{
    type : String,
    required : true,
    unique : true,
    lowercase:true,
    trim : true,
    index:true 
},
email:
{
    type:String,
    required:true,
    unique : true,
    lowercase:true,
    trim : true,
},
fullname:
{
    type:String,
    required:true,
    trim:true,
    index:true
},
avatar:
{
    type:String,//cloudnianry url
    required:true,

},

password:
{
    type:String,
    required:[true, 'password is requirred']


},
refreshtoken:{
type:String,
}



},
{
    timestamps: true
})

//save is an event
//tip dont use ()=>{} beacuse arrow func dont have refrence

userSchema.pre("save", async function(next){

  if(!this.isModified('password'))return next();
  this.password = await bcrypt.hash(this.password , 10)
  next();
})


userSchema.methods.isPasswordCorrect = async function(password)
{
return await bcrypt.compare(password ,this.password)
}

userSchema.methods.generateAccessToken=function(){
 return jwt.sign(
    {
        _id:this._id,
        email:this.email,
        fullname:this.fullname,
        username:this.username
    },process.env.ACCESS_TOKEN_SECRET,
    {
        expiresIn:process.env.ACCESS_TOKEN_EXPIRY
    }
 )

}
userSchema.methods.generateRefreshToken=function(){
    return jwt.sign(
        {
            _id:this._id,
            email:this.email,
            fullname:this.fullname,
            username:this.username
        },process.env.REFRESH_TOKEN_SECRET,
        {
            expiresIn:process.env.REFRESH_TOKEN_EXPIRY
        }
     )
}





//jwt aik beared token ha
export const User = mongoose.model("User",userSchema)