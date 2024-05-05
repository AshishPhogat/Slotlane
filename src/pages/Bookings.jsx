import React, { useContext, useEffect, useState } from "react";
import BookingCard from "../components/BookingCard";
import { UserContext } from "../Context/UserContextProvider";
import axios from "axios";

function Bookings() {
  const [bookings, setBookings] = useState([]);

  const { user } = useContext(UserContext);

  useEffect(() => {
    axios
      .get(`http://localhost:8000/bookings/${user.email}`)
      .then((res) => {
        setBookings(res.data.bookings);
      })
      .catch((error) => {
        console.log(error);
      });
  }, []);

  return (
    <>
      <div className="text-center">
        <h1 className="text-bold text-3xl text-blue-900 ">Bookings :</h1>
      </div>
      <div className="flex gap-2">
        {(!bookings)?null:(
          bookings.map((booking,idx)=>{
            return (
              <BookingCard key={idx} {...booking} />
            );
          }))
        }
      </div>
    </>
  );
}

export default Bookings;
