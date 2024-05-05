import React, { useContext } from "react";
import { UserContext } from "../Context/UserContextProvider";
import { signOut } from "firebase/auth";
import { auth } from "../firebase";
import { useNavigate } from "react-router-dom";
import { START_STATE } from "../Context/UserContextProvider";

function Credentials() {
  const { user,setUser } = useContext(UserContext);
  const navigate = useNavigate();
  const logout = () => {
    signOut(auth).then(() => {
      setUser(START_STATE);
      navigate("/");
    }).catch((error)=>{
        console.log(error);
    });
  };

  return (
    <>
      <p>username : {user.email}</p>
      <p className="bg-white text-center p-1">
        <button className="bg-blue-900 w-3/4 rounded-md" onClick={logout}>
          Logout
        </button>
      </p>
    </>
  );
}

export default Credentials;
