// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import {getAuth} from "firebase/auth";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBC0g9MsyNGv0YlEUU4dIQVUIDaKnoQFUw",
  authDomain: "slotlane-a1b82.firebaseapp.com",
  projectId: "slotlane-a1b82",
  storageBucket: "slotlane-a1b82.appspot.com",
  messagingSenderId: "1047133101741",
  appId: "1:1047133101741:web:8b6a97ff527f9b4c3a7024",
  measurementId: "G-JY1DVGJ67Z",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
const analytics = getAnalytics(app);

export default app;
