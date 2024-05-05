import React, { useContext, useEffect } from "react";
import { UserContext } from "../Context/UserContextProvider";
import Login from "../pages/Login";
import { useNavigate } from "react-router-dom";

function Protecting(props) {
  const { user } = useContext(UserContext);
  const navigate = useNavigate();

  useEffect(()=>{
    // console.log(user);
    if(!user.islogged ){
        navigate("/login"); 
    }
  },[user])


  return (
    <>
        {props.children}
    </>
  );
}

export default Protecting;
