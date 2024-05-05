import React, { useEffect, useState } from 'react'
import {auth} from "../firebase";
import { onAuthStateChanged, signOut } from 'firebase/auth';
import { Navigate, useNavigate } from 'react-router-dom';

export const UserContext = React.createContext();

export const START_STATE = {
    email : "",
    islogged : false
};

function UserContextProvider(props) {
    const [user,setUser] = useState(START_STATE);
    const navigate = useNavigate();
    useEffect(()=>{
        const unsubscribe = onAuthStateChanged(auth, (currentuser) => {
            // console.log(auth);
          });
      
          return () => {
            unsubscribe();
          };
    },[]);

  return (
    <>
        <UserContext.Provider value={{user,setUser}} >
            {props.children}
        </UserContext.Provider>
    </>
  )
}

export default UserContextProvider