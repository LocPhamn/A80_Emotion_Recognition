const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
  const Video = sequelize.define('Video', {
    idvideo: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true,
      field: 'idvideo'
    },
    video_name: {
      type: DataTypes.STRING(50),
      allowNull: true
    },
    zone_id: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    duration: {
      type: DataTypes.INTEGER,
      allowNull: true,
      comment: 'Duration in seconds'
    },
    date: {
      type: DataTypes.DATE,
      allowNull: true
    },
    file_path: {
      type: DataTypes.STRING(45),
      allowNull: true
    },
    status: {
      type: DataTypes.STRING(30),
      allowNull: true
    }
  }, {
    tableName: 'video',
    timestamps: false
  });

  Video.associate = (models) => {
    // Video has many Statistics
    Video.hasMany(models.Statistic, {
      foreignKey: 'video_id',
      as: 'statistics'
    });
  };

  return Video;
};
