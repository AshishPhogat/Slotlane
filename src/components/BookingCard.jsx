import React from "react";

const img =
  "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQNwcTTew6SrG-BDisWgqwfpLKfg8Ro8m1bTQ&usqp=CAU";

function BookingCard(props) {
  const { name, carNumber, phone } = {...props};
  return (
    <div className="p-2 bg-blue-900 w-1/4 rounded-md m-5 text-white">
      <div >
        <img className="rounded-md" src={img} />
      </div>
      <div>
        <p>Name : {name}</p>
        <p>Car Number : {carNumber}</p>
        <p>Phone : {phone}</p>
      </div>
    </div>
  );
}

export default BookingCard;
