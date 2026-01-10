import './LoginCard.css'

function LoginCard(){
    return(
        <div className="login-card">
            <h2>Welcome Back</h2>
            <div className="login-box">
                <p>Log in to your account</p>
                <form>
                    <label className='input-name' htmlFor="username">Username</label>
                    <input className="input-field" type="text" placeholder="Username" required />
                    <label className='input-name' htmlFor="password">Password</label>
                    <input className="input-field" type="password" placeholder="Password" required />
                </form>
            </div>
            <button type="submit" className="login-button">Log In</button>
            <p className="signup-link">Don't have an account? <a href="#">Sign Up</a></p>
        </div>
    )
}

export default LoginCard