import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import bagalogo from '../../../../images/bagalogo.png';
import user from '../../../../images/user.png';
import sets from '../../../../images/sets.png';
import vector from '../../../../images/Vector@2x.png';
import notif from '../../../../images/Vector3.png';
import patient from '../../../../images/Group@2x.png';
import message from '../../../../images/Vector4.png';
import contact from '../../../../images/Vector22x.png';
import tech from '../../../../images/Vector6.png';
import settings from '../../../../images/Vector5.png';
import toggle_btn from "../../../../images/toggle_btn.png";

const DoctorsSidebar = () => {
    const navigate = useNavigate();
    const location = useLocation(); 
    const [isSidebarVisible, setIsSidebarVisible] = useState(true);

    const username = localStorage.getItem('username') || 'User';

    const toggleSidebar = () => {
        setIsSidebarVisible(!isSidebarVisible);
    };

    const SidebarItem = ({ icon, label, path }) => {
        const isActive = location.pathname === path; 

        return (
            <div
                className={`flex items-center px-4 py-2 cursor-pointer rounded 
                ${isActive ? 'bg-[#005f6b]' : 'hover:bg-[#009099]'}`}
                onClick={() => navigate(path)}
            >
                <img src={icon} alt={label} className="object-contain" />
                {isSidebarVisible && <span className="ml-4 text-white">{label}</span>}
            </div>
        );
    };

    return (
        <div className={`relative flex flex-col ${isSidebarVisible ? 'w-64' : 'w-16'} bg-[#283342] h-full transition-all duration-300`}>
            <div className="absolute bottom-12 right-[-14px] z-10">
                <img
                    src={toggle_btn}
                    alt="Toggle Sidebar"
                    className="cursor-pointer"
                    onClick={toggleSidebar}
                />
            </div>

            <div className="flex items-center bg-[#1D242E] text-white p-4">
                <img src={bagalogo} alt="Baga" className={`object-contain ${isSidebarVisible ? 'h-5' : 'h-8'}`} />
                {isSidebarVisible && <span className="ml-2 font-bold text-xl">BAGA.NET</span>}
            </div>

            <div className="flex items-center p-4">
                <img src={user} alt="User" className="h-10 object-contain" />
                {isSidebarVisible && (
                    <div className="flex flex-col ml-2">
                        <span className="text-white font-medium text-lg">{username}</span> 
                        <span className="text-yellow-400 font-normal text-xs">Doctor</span>
                    </div>
                )}
                {isSidebarVisible && (
                    <img src={sets} alt="Settings" className="h-5 object-contain ml-auto cursor-pointer" />
                )}
            </div>

            <nav className="text-white text-base mt-4 ">
                <SidebarItem icon={vector} label="Dashboard" path="/doctors-dashboard" />
                <SidebarItem icon={patient} label="Patient Records" path="/doctors-patientActivity" />
                <SidebarItem icon={contact} label="Contact Management" path="/doctors-contactManagement" />
                <SidebarItem icon={notif} label="Notifications" path="/notifications" />
                <SidebarItem icon={message} label="Chat with Doctors" path="/chat-with-doctors" />
            </nav>


            <nav className="text-white text-base mt-auto mb-8">
                <hr className="border-zinc-500 mb-4" />
                <SidebarItem icon={settings} label="Settings" path="/settings" />
                <SidebarItem icon={tech} label="Get Technical Help" path="/technical-help" />
            </nav>

            <div className='bg-[#1D242E] p-4 text-center text-xs text-white'>
                Powered by BAGA.NET © 2024
            </div>
        </div>
    );
};

export default DoctorsSidebar;
