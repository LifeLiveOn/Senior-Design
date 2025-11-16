import CustomerTable from './components/CustomerTable'


function App() {
  return (
    <div>
      <div className='Header'>Roof Vision</div>
      <div className='MainBody'>
          <CustomerTable></CustomerTable>
      </div>
    </div>
  );
}

export default App;

// import logo from './logo.svg';
// import './App.css';
// import Home from "./pages/Home";

// function App() {
//   const [page, setPage] = useState("home")
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//         <h1>This is a test</h1>
//       </header>
//     </div>
//   );
// }

// export default App;