import './Header.css'


function Header(){
    return(
        <header>
            <h1 className="title">FormFit</h1>
            <button className="profile-button circle"><img className="profile-img" src="../public/profile.svg" alt="Profile"></img></button>
        </header>
    )
}

export default Header