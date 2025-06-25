import search from '../../../../images/search.png';
import eng from '../../../../images/eng.png';

const Navbar = () => {
  const currentDateTime = new Date().toLocaleString('en-US', {
    day: 'numeric', 
    month: 'long', 
    year: 'numeric',
    hour: '2-digit', 
    minute: '2-digit', 
    second: '2-digit'
  });

  return (
    <div className="flex items-center justify-between p-2 bg-white shadow-md px-6 ">
      <div className="flex items-center -ml-2">
        <input 
          type="text" 
          placeholder="Search for anything here..." 
          aria-label="Search bar" 
          className="border rounded-md px-4 py-2 w-96 ml-2 focus:outline-none focus:ring-1 bg-[#E3EBF3] focus:ring-blue-500" 
        />
        <img src={search} alt="Search icon" className="w-5 h-5 -ml-8 cursor-pointer mt-2" />
      </div>
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2">
          <img src={eng} alt="Language icon" className="w-5 h-5" />
          <span className="text-gray-600">English (US)</span>
        </div>
        <div className="flex items-center">
          <div className="text-yellow-500 font-semibold">●</div>
        </div>
        <div>
          <h1 className="text-gray-700 font-semibold">Good Afternoon</h1>
          <p className="text-sm text-gray-500">{currentDateTime}</p>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
