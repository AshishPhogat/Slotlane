import React, { useContext, useState } from "react";
import { auth } from "../firebase";
import { signInWithEmailAndPassword } from "firebase/auth";
import { UserContext } from "../Context/UserContextProvider";
import Navbar from "../components/Navbar";
import { useNavigate } from "react-router-dom";

const START_STATE = {
  email: "",
  password: "",
};

function Login() {
  const [details, setDetails] = useState(START_STATE);
  const [error, setError] = useState(false);
  const { user, setUser } = useContext(UserContext);
  const navigate = useNavigate();

  const userlogin = (e) => {
    e.preventDefault();
    if (details.email === "" || details.password === "") {
      alert("enter all the details carefully!!");
      return;
    }
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
        <h1 className="text-3xl">User Login</h1>
        <form className="text-center">
          <div className="my-2">
            <label htmlFor="email">Username/Email : </label>
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
          <div
            className={
              error ? "flex items-center justify-evenly" : "text-center "
            }
          >
            {error ? (
              <button
                onClick={() => {
                  navigate("/signup");
                }}
                className="bg-white px-2 rounded-md text-blue-900"
              >
                SignUp
              </button>
            ) : null}
            <button
              onClick={userlogin}
              className="bg-white px-2 rounded-md text-blue-900"
            >
              Login
            </button>
          </div>
        </form>
      </div>
    </>
  );
}

export default Login;
