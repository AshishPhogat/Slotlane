import { Routes, Route } from "react-router-dom";
import "./style.css";
import Login from "./pages/Login";
import Protecting from "./components/Protecting";
import Signup from "./pages/Signup";
import Homehoc from "./HOC/Homehoc";

function App() {

  return (
    <div className="App">
      <Routes>
        <Route exact path="/login" Component={Login} />
        <Route exact path="/signup" Component={Signup} />
        <Route exact path="/" Component={()=>{return (<Protecting><Homehoc/></Protecting>)}} />
      </Routes>
      
    </div>
  );
}

export default App;
