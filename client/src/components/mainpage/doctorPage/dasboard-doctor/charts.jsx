import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const data = [
  { name: 'Jan', patients: 30, reports: 15 },
  { name: 'Feb', patients: 45, reports: 20 },
  { name: 'Mar', patients: 35, reports: 25 },
  { name: 'Apr', patients: 50, reports: 30 },
];

const Charts = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-white p-6 rounded shadow-md">
        <h3 className="text-lg font-semibold mb-4">Patients Over Time</h3>
        <LineChart width={400} height={200} data={data}>
          <XAxis dataKey="name" />
          <YAxis />
          <CartesianGrid stroke="#e0dfdf" />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="patients" stroke="#8884d8" />
        </LineChart>
      </div>
      <div className="bg-white p-6 rounded shadow-md">
        <h3 className="text-lg font-semibold mb-4">Reports Generated</h3>
        <BarChart width={400} height={200} data={data}>
          <XAxis dataKey="name" />
          <YAxis />
          <CartesianGrid stroke="#e0dfdf" />
          <Tooltip />
          <Legend />
          <Bar dataKey="reports" fill="#82ca9d" />
        </BarChart>
      </div>
    </div>
  );
};

export default Charts;
