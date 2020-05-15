import React, { Component } from 'react';
import javascriptLogo from './JavaScript-logo.png';
import './App.css';

class App extends Component {
  render() {
    return (
      <div className="App">
        <img src={javascriptLogo} alt='Javascript logo'/>
        <h1>Welcome to Kodflix!</h1>
      </div>
    );
  }
}

export default App;
