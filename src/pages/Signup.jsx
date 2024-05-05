import React, { useContext, useState } from "react";
import { auth } from "../firebase";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
} from "firebase/auth";
import { UserContext } from "../Context/UserContextProvider";
import Navbar from "../components/Navbar";
import { useNavigate } from "react-router-dom";

const START_STATE = {
  email: "",
  password: "",
};

function Signup() {
  const [details, setDetails] = useState(START_STATE);
  const { user, setUser } = useContext(UserContext);
  const [error, setError] = useState("");

  const navigate = useNavigate();

  const usersignup = (e) => {
    e.preventDefault();
    if (details.email === "" || details.password == "") {
      alert("enter all the details carefully!!");
      return;
    }
    setError("");
    createUserWithEmailAndPassword(auth, details.email, details.password)
      .then((userCredentials) => {
        console.log(userCredentials);
        signInWithEmailAndPassword(auth, details.email, details.password)
          .then((userCredentials) => {
            setUser({
              email: userCredentials.user.email,
              islogged: true,
            });
            navigate("/");
          })
          .catch((error) => {
            setError(true);
          });
      })
      .catch((error) => {
        setError(error.message);
      });
  };

  const updateValue = (e) => {
    const field = e.target.id;
    setDetails((prev) => {
      return { ...prev, [field]: e.target.value };
    });
  };

  return (
    <>
      <Navbar />
      <div className="flex items-center justify-center flex-col bg-blue-900 w-3/4 rounded-xl h-64 text-white mx-auto mt-24 p-12">
        <h1 className="text-3xl">Sign Up</h1>
        <form className="text-center">
          <div className="my-2">
            <label htmlFor="email">Email : </label>
            <input
              className="rounded-md text-blue-500"
              onChange={updateValue}
              id="email"
              type="email"
              value={details.email}
              required
            />
            <br />
          </div>
          <div className="my-2">
            <label htmlFor="password">Password: </label>
            <input
              className="rounded-md text-blue-500"
              onChange={updateValue}
              id="password"
              type="password"
              value={details.password}
              required
            />
            <br />
          </div>
          <div className="text-center">
            <button
              onClick={usersignup}
              className="bg-white px-2 rounded-md text-blue-900"
            >
              Create Account
            </button>
          </div>
          {error == "" ? null : (
            <p className="text-md text-yellow-600">{error};</p>
          )}
        </form>
      </div>
    </>
  );
}

export default Signup;
