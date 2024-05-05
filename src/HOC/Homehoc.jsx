import React, { useContext } from 'react'
import Home from '../pages/Home'
import Bookings from '../pages/Bookings'
import Navbar from '../components/Navbar'
import { PageContext } from '../Context/PageContextProvider';

function Homehoc() {

    const { page, setPage } = useContext(PageContext);

  return (
    <>
      <Navbar />
      {
        (page=="Home")?(<Home/>):(<Bookings/>)
      }
    </>
  )
}

export default Homehoc