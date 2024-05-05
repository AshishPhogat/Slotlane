import React, { useContext, useState } from "react";
import { faUser } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { PageContext } from "../Context/PageContextProvider";
import { useNavigate } from "react-router-dom";
import { UserContext } from "../Context/UserContextProvider";
import Credentials from "./Credentials";

function Navbar() {
  const { setPage } = useContext(PageContext);
  const [display, setDisplay] = useState(false);
  const navigate = useNavigate();
  const { user, setUser } = useContext(UserContext);

  return (
    <div className="relative">
      {display ? (
        <div className="absolute bg-blue-900 text-white p-4 right-0 top-10">
          {(!user.islogged) ? (
            <>
              <p>
                <a href="/login">Login </a>/<a href="/signup"> SignUp</a>
              </p>
            </>
          ) : (
            <Credentials />
          )}
        </div>
      ) : null}
      <div className="flex bg-blue-900 text-white justify-between p-2">
        <div>SlotLane</div>
        <div className="flex gap-16">
          <a
            onClick={() => {
              setPage("Home");
              navigate("/");
            }}
          >
            Home
          </a>
          <a
            onClick={() => {
              setPage("Bookings");
              navigate("/");
            }}
          >
            Bookings
          </a>
          <a onClick={()=>{setDisplay(!display)}}>
            <FontAwesomeIcon icon={faUser} />{" "}
          </a>
        </div>
      </div>
    </div>
  );
}

export default Navbar;
