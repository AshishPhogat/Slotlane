import React, { useContext, useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import axios from "axios";
import { UserContext } from "../Context/UserContextProvider";

const background =
  "https://media.istockphoto.com/id/1356830579/photo/newly-striped-parking-lot.jpg?s=612x612&w=0&k=20&c=oJtKltmJGjsZ5SVllCVwm3cJzHQafcZmgbmbpWkH3tw=";

// const START_DETAILS = {
//   name: "",
//   carNumber: "",
//   phone: 0,
// };

function Home() {
  const [booking, setBooking] = useState(false);
  const [slots, setSlots] = useState(0);
  const [details, setDetails] = useState({
    name: "",
    carNumber: "",
    phone: 0,
  });
  const { user } = useContext(UserContext);

  useEffect(() => {
    axios.get("http://localhost:8000/slots").then((res)=>{
      console.log(slots);
      setSlots(res.data.slots);
    });
  }, [details]);

  const BookSlot = async (e) => {
    e.preventDefault();
    if (!details.carNumber || !details.name || !details.phone) {
      alert("please enter the values in all fields");
      return;
    }
    await axios
      .post(`http://localhost:8000/slots/${user.email}`, details)
      .then((response) => {
        console.log(response.data);
        setSlots(response.data.slots);
      })
      .catch((error) => {
        console.log(error.message);
      });
      
      setDetails({
        name: "",
        carNumber: "",
        phone: 0,
      });

  };

  const updateValue = (e) => {
    const field = e.target.id;
    setDetails({ ...details, [field]: e.target.value });
  };

  return (
    <div>
      <div className="h-dvh w-dvw p-12">
        <div className="text-center">
          <div className="text-center border-black border-2 rounded-md">
            Available Slots : {slots}
          </div>
          <button
            onClick={() => {
              setBooking(!booking);
            }}
            className="text-white bg-blue-900 p-2 rounded-md mt-8"
          >
            {booking ? "Cancel" : "Book Slot"}
          </button>
        </div>
        {!booking ? null : slots === 0 ? (
          <div id="warning" className="text-red-700 text-center mt-5">
            No Slots Available
          </div>
        ) : (
          <form className="text-blue-900 m-auto border-2 w-1/2 p-10">
            <label htmlFor="name">Name :</label>
            <br />
            <input
              onChange={updateValue}
              value={details.name}
              className="text-black"
              id="name"
              type="text"
              required
            />
            <br />
            <label htmlFor="carNumber"> Car Number :</label>
            <br />
            <input
              onChange={updateValue}
              value={details.carNumber}
              className="text-black"
              id="carNumber"
              type="text"
              required
            />
            <br />
            <label htmlFor="phone">Phone Number : </label>
            <br />
            <input
              onChange={updateValue}
              value={details.phone}
              className="text-black"
              id="phone"
              type="tel"
              required
            />
            <br />
            <button
              onClick={BookSlot}
              className="bg-blue-900 text-white rounded-md p-2"
            >
              Book Slot
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

export default Home;
